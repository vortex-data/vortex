use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use dashmap::DashMap;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_expr::transform::partition::{PartitionedExpr, partition};

use crate::layouts::struct_::StructLayout;
use crate::{
    ArrayEvaluation, LayoutChildren, LayoutReader, LayoutReaderRef, LazyReaderChildren,
    MaskEvaluation, PruningEvaluation,
};

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RowIdLayout {
    row_count: u64,
    child: LayoutReaderRef,
}

pub struct RowIdLayoutReader {
    layout: RowIdLayout,
    name: Arc<str>,
    partitioned_expr_cache: DashMap<ExactExpr, Arc<PartitionedExpr>>,
}

impl RowIdLayoutReader {
    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: ExprRef) -> Arc<PartitionedExpr> {
        self.partitioned_expr_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // Partition the expression into expressions that can be evaluated over individual fields
                Arc::new(
                    partition(expr, self.dtype()).vortex_expect(
                        "We should not fail to partition expression over struct fields",
                    ),
                )
            })
            .clone()
    }
}

impl LayoutReader for RowIdLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.child.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count)
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.layout
            .child
            .register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        todo!()
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        todo!()
    }
}
