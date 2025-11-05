// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask, FieldName, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_expr::transform::immediate_access::annotate_scope_access;
use vortex_expr::transform::{
    PartitionedExpr, partition, replace, replace_root_fields, simplify_typed,
};
use vortex_expr::{ExactExpr, Expression, col, root};
use vortex_gpu::{GpuStructVector, GpuVector};
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::gpu::children::LazyGpuReaderChildren;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSource;
use crate::{GpuArrayFuture, GpuLayoutReader, GpuLayoutReaderRef};

pub struct GpuStructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyGpuReaderChildren,
    ctx: Arc<CudaContext>,

    /// A `pack` expression that holds each individual field of the root DType. This expansion
    /// ensures we can correctly partition expressions over the fields of the struct.
    expanded_root_expr: Expression,

    field_lookup: Option<HashMap<FieldName, usize>>,
    partitioned_expr_cache: DashMap<ExactExpr, Partitioned>,
}

impl GpuStructReader {
    pub(crate) fn try_new(
        layout: StructLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: Arc<CudaContext>,
    ) -> VortexResult<Self> {
        let struct_dt = layout.struct_fields();

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of field names
        let field_lookup = (struct_dt.nfields() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect()
        });

        let lazy_children =
            LazyGpuReaderChildren::new(layout.children().clone(), segment_source.clone());

        // Create an expanded root expression that contains all fields of the struct.
        let expanded_root_expr = replace_root_fields(root(), struct_dt);

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            name,
            expanded_root_expr,
            lazy_children,
            ctx,
            field_lookup,
            partitioned_expr_cache: Default::default(),
        })
    }

    /// Return the [`StructFields`] of this layout.
    fn struct_fields(&self) -> &StructFields {
        self.layout.struct_fields()
    }

    /// Return the child reader for the field.
    fn child(&self, name: &FieldName) -> VortexResult<&GpuLayoutReaderRef> {
        let idx = self
            .field_lookup
            .as_ref()
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.struct_fields().find(name))
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;
        self.child_by_idx(idx)
    }

    /// Return the child reader for the field, by index.
    fn child_by_idx(&self, idx: usize) -> VortexResult<&GpuLayoutReaderRef> {
        let field_dtype = self
            .struct_fields()
            .field_by_index(idx)
            .ok_or_else(|| vortex_err!("Missing field {idx}"))?;
        let name = &self.struct_fields().names()[idx];
        self.lazy_children.get(
            idx,
            &field_dtype,
            &format!("{}.{}", self.name, name).into(),
            &self.ctx,
        )
    }

    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: Expression) -> Partitioned {
        self.partitioned_expr_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // First, we expand the root scope into the fields of the struct to ensure
                // that partitioning works correctly.
                let expr = replace(expr.clone(), &root(), self.expanded_root_expr.clone());
                let expr = simplify_typed(expr, self.dtype())
                    .vortex_expect("We should not fail to simplify expression over struct fields");

                // Partition the expression into expressions that can be evaluated over individual fields
                let mut partitioned = partition(
                    expr.clone(),
                    self.dtype(),
                    annotate_scope_access(
                        self.dtype()
                            .as_struct_fields_opt()
                            .vortex_expect("We know it's a struct DType"),
                    ),
                )
                .vortex_expect("We should not fail to partition expression over struct fields");

                if partitioned.partitions.len() == 1 {
                    // If there's only one partition, we step into the field scope of the original
                    // expression by replacing any `$.a` with `$`.
                    return Partitioned::Single(
                        partitioned.partition_names[0].clone(),
                        replace(expr, &col(partitioned.partition_names[0].clone()), root()),
                    );
                }

                // We now need to process the partitioned expressions to rewrite the root scope
                // to be that of the field, rather than the struct. In other words, "stepping in"
                // to the field scope.
                partitioned.partitions = partitioned
                    .partitions
                    .iter()
                    .zip_eq(partitioned.partition_names.iter())
                    .map(|(e, name)| replace(e.clone(), &col(name.clone()), root()))
                    .collect();

                Partitioned::Multi(Arc::new(partitioned))
            })
            .clone()
    }
}

/// When partitioning an expression, in the case it only has a single partition we can avoid
/// some cost and just delegate to the child reader directly.
#[derive(Clone)]
enum Partitioned {
    Single(FieldName, Expression),
    Multi(Arc<PartitionedExpr<FieldName>>),
}

impl GpuLayoutReader for GpuStructReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // In the case of an empty struct, we need to register the end split.
        splits.insert(row_offset + self.layout.row_count());

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.child_by_idx(idx)?
                .register_splits(&[mask], row_offset, splits)
        })
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
    ) -> VortexResult<GpuArrayFuture> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let len = usize::try_from(row_range.end - row_range.start)
            .vortex_expect("read range len must fit into usize");
        match &self.partition_expr(expr.clone()) {
            Partitioned::Single(name, partition) => self
                .child(name)?
                .projection_evaluation(row_range, partition),
            Partitioned::Multi(partitioned) => {
                let partitioned = partitioned.clone();
                // Construct evaluations for each child.
                let field_evals: Vec<_> = partitioned
                    .partition_annotations
                    .iter()
                    .zip_eq(partitioned.partitions.iter())
                    .map(|(annotation, expr)| {
                        self.child(annotation)?
                            .projection_evaluation(row_range, expr)
                    })
                    .try_collect()?;

                Ok(Box::pin(async move {
                    // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
                    let field_arrays = try_join_all(field_evals).await?;

                    Ok(vec![GpuVector::Struct(GpuStructVector::new(
                        partitioned.partition_names.clone(),
                        field_arrays.into(),
                        len,
                    ))])
                }))
            }
        }
    }
}
