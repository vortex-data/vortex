// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::IntoArray;
use vortex_array::ProstMetadata;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layouts::dict::DictLayoutMetadata;
use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::planner::ComputeArgs;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeOp;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The dictionary layout vtable.
#[derive(Clone)]
pub struct Dict;

/// Metadata for a dictionary layout.
///
/// Stores the child dtypes (codes and values) and the `all_values_referenced` optimization hint.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DictV2Metadata {
    /// DType for values (child 0) and codes (child 1), matching the v1 child order.
    child_dtypes: [DType; 2],
    /// Whether all dictionary values are referenced by at least one code.
    all_values_referenced: bool,
}

impl fmt::Display for DictV2Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DictMetadata(all_values_referenced={})",
            self.all_values_referenced
        )
    }
}

/// Prost-compatible metadata matching the v1 `DictLayoutMetadata` wire format.
#[derive(prost::Message)]
struct DictLayoutMetadataProto {
    #[prost(enumeration = "PType", tag = "1")]
    codes_ptype: i32,
    #[prost(optional, bool, tag = "2")]
    is_nullable_codes: Option<bool>,
    #[prost(optional, bool, tag = "3")]
    all_values_referenced: Option<bool>,
}

impl LayoutVTable for Dict {
    type Metadata = DictV2Metadata;

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.dict")
    }

    fn deserialize_metadata(
        metadata: &[u8],
        dtype: &DType,
        _row_count: u64,
        _children: &[LayoutChild],
        _array_ctx: &ReadContext,
    ) -> VortexResult<Self::Metadata> {
        let proto = ProstMetadata::<DictLayoutMetadata>::deserialize(metadata)?;

        let codes_ptype = proto.codes_ptype();
        let codes_nullable = Nullability::from(proto.is_nullable_codes());
        let all_values_referenced = proto.all_values_referenced();

        let codes_dtype = DType::Primitive(codes_ptype, codes_nullable);
        // Values carry the parent dtype (the dict dtype = values dtype union codes nullability).
        let values_dtype = dtype.clone();

        // Child order matches v1: child 0 = values, child 1 = codes.
        Ok(DictV2Metadata {
            child_dtypes: [values_dtype, codes_dtype],
            all_values_referenced,
        })
    }

    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType {
        &layout.metadata().child_dtypes[child_idx]
    }

    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship {
        match child_idx {
            // Values (child 0): auxiliary data in a separate row space.
            0 => ChildRelationship::Auxiliary(0..layout.row_count()),
            // Codes (child 1): same row space as parent.
            1 => ChildRelationship::RowOffset(0),
            _ => unreachable!("Dict layout has only 2 children, got index {child_idx}"),
        }
    }

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &Selection,
        row_offset: Option<u64>,
        row_splits: &mut BTreeSet<u64>,
        session: &VortexSession,
    ) -> VortexResult<SplitPlannerRef> {
        let codes_child = layout.codes_child()?;
        let values_child = layout.values_child()?;
        let values_row_count = values_child.row_count();
        let values_rel = Self::child_relationship(layout, 0);
        let codes_rel = Self::child_relationship(layout, 1);

        // Only the codes child contributes row split boundaries.
        let codes_offset = codes_rel.child_row_offset(row_offset);
        let codes_planner =
            codes_child.prepare(&root(), selection, codes_offset, row_splits, session)?;

        // Values are auxiliary — they don't register split boundaries.
        let values_offset = values_rel.child_row_offset(row_offset); // always None
        let values_planner =
            values_child.prepare(&root(), selection, values_offset, row_splits, session)?;

        Ok(Arc::new(DictSplitPlanner {
            expression: expr.clone(),
            all_values_referenced: layout.metadata().all_values_referenced,
            codes_planner,
            codes_relationship: codes_rel,
            values_planner,
            values_row_count,
            values_relationship: values_rel,
        }))
    }
}

impl Layout<Dict> {
    /// Returns the values child (child 0).
    pub fn values_child(&self) -> VortexResult<LayoutRef> {
        self.child(0)
    }

    /// Returns the codes child (child 1).
    pub fn codes_child(&self) -> VortexResult<LayoutRef> {
        self.child(1)
    }
}

struct DictSplitPlanner {
    expression: Expression,
    all_values_referenced: bool,
    codes_planner: SplitPlannerRef,
    codes_relationship: ChildRelationship,
    values_planner: SplitPlannerRef,
    values_row_count: u64,
    values_relationship: ChildRelationship,
}

impl SplitPlanner for DictSplitPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        // Plan codes in the parent's row space.
        let codes_output = {
            let mut codes_builder = builder.step_into(&self.codes_relationship);
            self.codes_planner
                .plan_split(row_range, selection, &mut codes_builder)?
        };

        // Plan values in the auxiliary row space.
        // Values are read in full (0..values_row_count) and shared across splits.
        let values_row_count = self.values_row_count;
        let values_output = {
            let mut values_builder = builder.step_into(&self.values_relationship);
            // Values are read unconditionally with an all-true selection mask.
            let values_selection = values_builder.create_node_resolved(
                #[allow(clippy::cast_possible_truncation)]
                vortex_mask::Mask::AllTrue(values_row_count as usize).into_array(),
                0..values_row_count,
            );
            self.values_planner.plan_split(
                &(0..values_row_count),
                values_selection,
                &mut values_builder,
            )?
        };

        // Combine codes + values into a DictArray and apply the expression.
        let expression = self.expression.clone();
        let all_values_referenced = self.all_values_referenced;
        builder.create_node(NodeOpts {
            op: NodeOp::Custom { label: "Dict" },
            inputs: &[codes_output, values_output],
            segments: vec![],
            lifetime: builder.row_range_lifetime(row_range.clone()),
            compute: move |args: ComputeArgs| {
                let mut inputs = args.inputs.into_iter();
                let codes = inputs.next().vortex_expect("missing codes");
                let values = inputs.next().vortex_expect("missing values");

                // SAFETY: Layout was validated at write time.
                let array = unsafe {
                    DictArray::new_unchecked(codes, values)
                        .set_all_values_referenced(all_values_referenced)
                }
                .into_array()
                .optimize()?;

                array.apply(&expression)
            },
        })
    }
}
