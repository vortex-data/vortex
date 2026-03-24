// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::make_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::layout::RowSelection;
use crate::v2::planner::NodeId;
use crate::v2::planner::NodeInput;
use crate::v2::planner::NodeOpts;
use crate::v2::planner::PlanBuilder;
use crate::v2::planner::SplitPlanner;
use crate::v2::planner::SplitPlannerRef;
use crate::v2::planner::SplitSelection;

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
    type Plan = ();

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.struct")
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
        selection: &RowSelection,
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef> {
        let struct_fields = layout.dtype().as_struct_fields();

        // Partition the expression over struct fields using the same approach as legacy
        // StructReader.
        let partitioned = compute_partitioned_expr(expr, layout.dtype(), struct_fields);

        match partitioned {
            Partitioned::Single(field_name, field_expr) => {
                let Some(field_idx) = struct_fields.find(&field_name) else {
                    vortex_bail!("Partitioned field {field_name} not found in struct fields")
                };
                let child = layout.field_child(field_idx)?;
                let planner = child.prepare(&field_expr, selection, row_splits)?;

                // If nullable, also prepare validity.
                let validity_planner = if let Some(validity_child) = layout.validity_child()? {
                    Some(validity_child.prepare(&root(), selection, row_splits)?)
                } else {
                    None
                };

                Ok(Arc::new(SingleFieldSplitPlanner {
                    planner,
                    validity_planner,
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
                    let planner = child.prepare(partition_expr, selection, row_splits)?;
                    field_planners.push(planner);
                }

                // If nullable, also prepare validity.
                let validity_planner = if let Some(validity_child) = layout.validity_child()? {
                    Some(validity_child.prepare(&root(), selection, row_splits)?)
                } else {
                    None
                };

                Ok(Arc::new(MultiFieldSplitPlanner {
                    root_expr: partitioned_expr.root.clone(),
                    field_planners,
                    validity_planner,
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
    validity_planner: Option<SplitPlannerRef>,
}

impl SplitPlanner for SingleFieldSplitPlanner {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let data_output = self
            .planner
            .plan_split(row_range.clone(), selection, builder)?;

        let Some(validity_planner) = &self.validity_planner else {
            return Ok(data_output);
        };

        let validity_output = validity_planner.plan_split(row_range.clone(), selection, builder)?;

        builder.create_node(NodeOpts {
            inputs: &[data_output, validity_output],
            segments: &[],
            lifetime: builder.row_range_lifetime(row_range),
            compute: move |mut inputs: Vec<NodeInput>| {
                let _data = inputs.remove(0).into_array();
                let _validity = inputs.remove(0).into_array();
                todo!("apply validity mask to single-field data output")
            },
        })
    }
}

/// Split planner for a multi-field struct expression.
struct MultiFieldSplitPlanner {
    root_expr: Expression,
    field_planners: Vec<SplitPlannerRef>,
    validity_planner: Option<SplitPlannerRef>,
}

impl SplitPlanner for MultiFieldSplitPlanner {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let mut child_outputs = Vec::with_capacity(self.field_planners.len());
        for planner in &self.field_planners {
            let output = planner.plan_split(row_range.clone(), selection, builder)?;
            child_outputs.push(output);
        }

        if let Some(validity_planner) = &self.validity_planner {
            let validity_output =
                validity_planner.plan_split(row_range.clone(), selection, builder)?;
            child_outputs.push(validity_output);
        }

        let root_expr = self.root_expr.clone();
        let has_validity = self.validity_planner.is_some();
        builder.create_node(NodeOpts {
            inputs: &child_outputs,
            segments: &[],
            lifetime: builder.row_range_lifetime(row_range),
            compute: move |mut inputs: Vec<NodeInput>| {
                let validity = if has_validity {
                    inputs.pop().map(NodeInput::into_array)
                } else {
                    None
                };
                let field_arrays: Vec<ArrayRef> =
                    inputs.into_iter().map(|i| i.into_array()).collect();
                // TODO: pack field_arrays into a StructArray, apply validity,
                // then evaluate root_expr on the result.
                let _root_expr = root_expr;
                let _validity = validity;
                let _field_arrays = field_arrays;
                todo!("assemble struct from field arrays and evaluate root expression")
            },
        })
    }
}
