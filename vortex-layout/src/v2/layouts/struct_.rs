// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::make_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;
use vortex_array::scalar_fn::fns::merge::Merge;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::planner::ComputeArgs;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The struct layout vtable.
#[derive(Clone)]
pub struct Struct;

/// Metadata for a struct layout.
///
/// Stores pre-resolved child dtypes so `child_dtype` can return references.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructMetadata {
    /// Resolved dtypes for each child layout. If nullable, index 0 is the validity dtype,
    /// and data field dtypes start at index 1.
    pub child_dtypes: Vec<DType>,
}

impl fmt::Display for StructMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StructMetadata({} children)", self.child_dtypes.len())
    }
}

impl Layout<Struct> {
    /// Returns the child index offset for data fields.
    /// If the struct is nullable, child 0 is the validity layout, and data fields start at 1.
    fn data_child_offset(&self) -> usize {
        if self.dtype().is_nullable() { 1 } else { 0 }
    }

    /// Returns the layout child for the given field index (0-based field index).
    fn field_child(&self, field_idx: usize) -> VortexResult<crate::v2::layout::LayoutRef> {
        self.child(field_idx + self.data_child_offset())
    }

    /// Returns the child relationship for the given field index (0-based field index).
    fn field_child_relationship(&self, field_idx: usize) -> ChildRelationship {
        Struct::child_relationship(self, field_idx + self.data_child_offset())
    }

    /// Returns the validity child layout, if the struct is nullable.
    fn validity_child(&self) -> VortexResult<Option<crate::v2::layout::LayoutRef>> {
        if self.dtype().is_nullable() {
            Ok(Some(self.child(0)?))
        } else {
            Ok(None)
        }
    }
}

impl LayoutVTable for Struct {
    type Metadata = StructMetadata;

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.struct")
    }

    fn deserialize_metadata(
        metadata: &[u8],
        dtype: &DType,
        row_count: u64,
        children: &[LayoutChild],
        array_ctx: &ReadContext,
    ) -> VortexResult<Self::Metadata> {
        let struct_fields = dtype.as_struct_fields();
        let mut child_dtypes = Vec::new();
        if dtype.is_nullable() {
            // Child 0 is the validity layout (boolean non-nullable).
            child_dtypes.push(DType::Bool(Nullability::NonNullable));
        }
        for i in 0..struct_fields.nfields() {
            let name = struct_fields
                .field_name(i)
                .vortex_expect("Struct field index out of bounds");
            let field_dtype = struct_fields
                .field(name)
                .vortex_expect("Struct field not found");
            child_dtypes.push(field_dtype);
        }
        Ok(StructMetadata { child_dtypes })
    }

    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType {
        // FIXME(ngates): this should return VortexResult<DType> so we can traverse into
        //  children of lazy struct dtype.
        &layout.metadata().child_dtypes[child_idx]
    }

    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship {
        let struct_fields = layout.dtype().as_struct_fields();

        if layout.dtype().is_nullable() {
            if child_idx == 0 {
                ChildRelationship::RowOffset(0)
            } else {
                ChildRelationship::FieldName(
                    struct_fields
                        .field_name(child_idx - 1)
                        .vortex_expect("Struct field index out of bounds")
                        .clone(),
                )
            }
        } else {
            ChildRelationship::FieldName(
                struct_fields
                    .field_name(child_idx)
                    .vortex_expect("Struct field index out of bounds")
                    .clone(),
            )
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
        let struct_fields = layout.dtype().as_struct_fields();

        // Partition the expression over struct fields using the same approach as legacy
        // StructReader.
        let partitioned = compute_partitioned_expr(expr, layout.dtype(), struct_fields);

        // Pre-compute validity relationship if nullable.
        let validity_relationship = if layout.dtype().is_nullable() {
            Some(Self::child_relationship(layout, 0))
        } else {
            None
        };

        match partitioned {
            Partitioned::Single(field_name, field_expr) => {
                let Some(field_idx) = struct_fields.find(&field_name) else {
                    vortex_bail!("Partitioned field {field_name} not found in struct fields")
                };
                let child = layout.field_child(field_idx)?;
                let field_relationship = layout.field_child_relationship(field_idx);
                let child_offset = field_relationship.child_row_offset(row_offset);
                let planner =
                    child.prepare(&field_expr, selection, child_offset, row_splits, session)?;

                // If nullable, also prepare validity.
                let validity_planner = if let (Some(validity_child), Some(validity_rel)) =
                    (layout.validity_child()?, &validity_relationship)
                {
                    let val_offset = validity_rel.child_row_offset(row_offset);
                    Some(validity_child.prepare(
                        &root(),
                        selection,
                        val_offset,
                        row_splits,
                        session,
                    )?)
                } else {
                    None
                };

                let is_pack_merge = field_expr.is::<Pack>() || field_expr.is::<Merge>();

                Ok(Arc::new(SingleFieldSplitPlanner {
                    planner,
                    field_relationship,
                    validity_planner,
                    validity_relationship,
                    is_pack_merge,
                }))
            }
            Partitioned::Multi(partitioned_expr) => {
                let mut field_planners = Vec::with_capacity(partitioned_expr.partitions.len());

                for (partition_expr, annotation) in partitioned_expr
                    .partitions
                    .iter()
                    .zip(partitioned_expr.partition_annotations.iter())
                {
                    let Some(field_idx) = struct_fields.find(annotation) else {
                        vortex_bail!("Partitioned field {annotation} not found in struct fields")
                    };
                    let child = layout.field_child(field_idx)?;
                    let relationship = layout.field_child_relationship(field_idx);
                    let child_offset = relationship.child_row_offset(row_offset);
                    let planner = child.prepare(
                        partition_expr,
                        selection,
                        child_offset,
                        row_splits,
                        session,
                    )?;
                    field_planners.push((planner, relationship));
                }

                // If nullable, also prepare validity.
                let validity_planner = if let (Some(validity_child), Some(validity_rel)) =
                    (layout.validity_child()?, &validity_relationship)
                {
                    let val_offset = validity_rel.child_row_offset(row_offset);
                    Some(validity_child.prepare(
                        &root(),
                        selection,
                        val_offset,
                        row_splits,
                        session,
                    )?)
                } else {
                    None
                };

                let is_pack_merge =
                    partitioned_expr.root.is::<Pack>() || partitioned_expr.root.is::<Merge>();

                Ok(Arc::new(MultiFieldSplitPlanner {
                    root_expr: partitioned_expr.root.clone(),
                    partition_names: partitioned_expr.partition_names.clone(),
                    is_pack_merge,
                    field_planners,
                    validity_planner,
                    validity_relationship,
                }))
            }
        }
    }
}

/// Result of partitioning an expression over struct fields.
#[derive(Clone)]
enum Partitioned {
    /// Expression operates over a single field.
    Single(FieldName, Expression),
    /// Expression operates over multiple fields.
    Multi(Arc<vortex_array::expr::transform::PartitionedExpr<FieldName>>),
}

/// Partition an expression over struct fields following the same approach as legacy StructReader.
fn compute_partitioned_expr(
    expr: &Expression,
    dtype: &DType,
    struct_fields: &vortex_array::dtype::StructFields,
) -> Partitioned {
    use vortex_error::VortexExpect;

    // Step 1: expand root() → pack(a: $.a, b: $.b, c: $.c, …)
    let expanded = replace_root_fields(root(), struct_fields);
    let expr = replace(expr.clone(), &root(), expanded);
    let expr = expr
        .optimize_recursive(dtype)
        .vortex_expect("Failed to simplify expression over struct fields");

    // Step 2: partition into per-field sub-expressions
    let mut partitioned = partition(
        expr.clone(),
        dtype,
        make_free_field_annotator(struct_fields),
    )
    .vortex_expect("Failed to partition expression over struct fields");

    // Step 3a: single-partition fast path — rewrite $.field → $
    if partitioned.partitions.len() == 1 {
        return Partitioned::Single(
            partitioned.partition_names[0].clone(),
            replace(expr, &col(partitioned.partition_names[0].clone()), root()),
        );
    }

    // Step 3b: multi-partition — rewrite $.field_name → $ in each partition
    partitioned.partitions = partitioned
        .partitions
        .iter()
        .zip(partitioned.partition_names.iter())
        .map(|(e, name)| replace(e.clone(), &col(name.clone()), root()))
        .collect();

    Partitioned::Multi(Arc::new(partitioned))
}

/// Split planner for a single-field struct expression.
struct SingleFieldSplitPlanner {
    planner: SplitPlannerRef,
    field_relationship: ChildRelationship,
    validity_planner: Option<SplitPlannerRef>,
    validity_relationship: Option<ChildRelationship>,
    /// Whether the field expression produces a struct (Pack/Merge at top level).
    /// When true, validity is applied per-field to avoid changing the outermost struct's
    /// nullability (which may belong to a parent layout's wrapping expression).
    is_pack_merge: bool,
}

impl SplitPlanner for SingleFieldSplitPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let data_output = {
            let mut child_builder = builder.step_into(&self.field_relationship);
            self.planner
                .plan_split(row_range, selection, &mut child_builder)?
        };

        let Some(validity_planner) = &self.validity_planner else {
            return Ok(data_output);
        };

        let validity_output = {
            let mut child_builder = builder.step_into(self.validity_relationship.as_ref().unwrap());
            validity_planner.plan_split(row_range, selection, &mut child_builder)?
        };

        let is_pack_merge = self.is_pack_merge;
        builder.create_node(NodeOpts {
            label: "StructSingle",
            inputs: &[data_output, validity_output],
            segments: vec![],
            lifetime: builder.row_range_lifetime(row_range.clone()),
            compute: move |args: ComputeArgs| {
                let mut inputs = args.inputs.into_iter();
                let data = inputs.next().vortex_expect("missing");
                let validity = inputs.next().vortex_expect("missing");
                if is_pack_merge {
                    // Apply validity per-field to preserve the outer struct's nullability.
                    let struct_array = data.to_struct();
                    let masked_fields: Vec<ArrayRef> = struct_array
                        .unmasked_fields()
                        .iter()
                        .map(|a| a.clone().mask(validity.clone()))
                        .collect::<VortexResult<_>>()?;
                    Ok(StructArray::try_new(
                        struct_array.names().clone(),
                        masked_fields,
                        struct_array.len(),
                        struct_array.validity()?,
                    )?
                    .into_array())
                } else {
                    data.mask(validity)
                }
            },
        })
    }
}

/// Split planner for a multi-field struct expression.
struct MultiFieldSplitPlanner {
    root_expr: Expression,
    partition_names: FieldNames,
    is_pack_merge: bool,
    field_planners: Vec<(SplitPlannerRef, ChildRelationship)>,
    validity_planner: Option<SplitPlannerRef>,
    validity_relationship: Option<ChildRelationship>,
}

impl SplitPlanner for MultiFieldSplitPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let mut child_outputs = Vec::with_capacity(self.field_planners.len());
        for (planner, relationship) in &self.field_planners {
            let mut child_builder = builder.step_into(relationship);
            let output = planner.plan_split(row_range, selection, &mut child_builder)?;
            child_outputs.push(output);
        }

        if let Some(validity_planner) = &self.validity_planner {
            let mut child_builder = builder.step_into(self.validity_relationship.as_ref().unwrap());
            let validity_output =
                validity_planner.plan_split(row_range, selection, &mut child_builder)?;
            child_outputs.push(validity_output);
        }

        let root_expr = self.root_expr.clone();
        let partition_names = self.partition_names.clone();
        let is_pack_merge = self.is_pack_merge;
        let has_validity = self.validity_planner.is_some();
        builder.create_node(NodeOpts {
            label: "StructMulti",
            inputs: &child_outputs,
            segments: vec![],
            lifetime: builder.row_range_lifetime(row_range.clone()),
            compute: move |mut args: ComputeArgs| {
                let validity = if has_validity {
                    args.inputs.pop()
                } else {
                    None
                };
                let len = args.inputs.first().map_or(0, |a| a.len());

                // Assemble a StructArray from the field arrays.
                let root_scope =
                    StructArray::try_new(partition_names, args.inputs, len, Validity::NonNullable)?
                        .into_array();

                // Evaluate the root expression on the assembled struct.
                let result = root_scope.apply(&root_expr)?;

                // Apply validity if the struct is nullable.
                if let Some(validity) = validity {
                    if is_pack_merge {
                        // For pack/merge, apply validity per-field.
                        let struct_array = result.to_struct();
                        let masked_fields: Vec<ArrayRef> = struct_array
                            .unmasked_fields()
                            .iter()
                            .map(|a| a.clone().mask(validity.clone()))
                            .collect::<VortexResult<_>>()?;
                        Ok(StructArray::try_new(
                            struct_array.names().clone(),
                            masked_fields,
                            struct_array.len(),
                            struct_array.validity()?,
                        )?
                        .into_array())
                    } else {
                        result.mask(validity)
                    }
                } else {
                    Ok(result)
                }
            },
        })
    }
}
