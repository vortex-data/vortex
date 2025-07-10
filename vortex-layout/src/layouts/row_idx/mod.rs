// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
pub use expr::*;
use vortex_array::compute::filter;
use vortex_array::stats::Precision;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldMask, PType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::partition::{PartitionedExpr, partition};
use vortex_expr::transform::replace::replace;
use vortex_expr::{ExactExpr, ExprRef, Scope, ScopeDType, is_root, root};
use vortex_mask::Mask;
use vortex_scalar::PValue;
use vortex_sequence::SequenceArray;

use crate::layouts::partitioned::{PartitionedArrayEvaluation, PartitionedMaskEvaluation};
use crate::{
    ArrayEvaluation, LayoutReader, MaskEvaluation, NoOpMaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

pub struct RowIdxLayoutReader {
    name: Arc<str>,
    row_offset: u64,
    child: Arc<dyn LayoutReader>,

    partition_cache: DashMap<ExactExpr, Partitioning>,
}

impl RowIdxLayoutReader {
    pub fn new(row_offset: u64, child: Arc<dyn LayoutReader>) -> Self {
        Self {
            name: child.name().clone(),
            row_offset,
            child,
            partition_cache: DashMap::new(),
        }
    }

    fn partition_expr(&self, expr: &ExprRef) -> Partitioning {
        self.partition_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // Partition the expression into row idx and child expressions.
                let mut partitioned = partition(expr.clone(), self.dtype(), |expr| {
                    if expr.is::<RowIdxVTable>() {
                        vec![Partition::RowIdx]
                    } else if is_root(expr) {
                        vec![Partition::Child]
                    } else {
                        vec![]
                    }
                })
                .vortex_expect("We should not fail to partition expression over struct fields");

                // If there's only a single partition, we can directly return the expression.
                if partitioned.partitions.len() == 1 {
                    return match &partitioned.partition_annotations[0] {
                        Partition::RowIdx => {
                            Partitioning::RowIdx(replace(expr.clone(), &row_idx(), root()))
                        }
                        Partition::Child => Partitioning::Child(expr.clone()),
                    };
                }

                // Replace the row_idx expression with the root expression in the row_idx partition.
                partitioned.partitions = partitioned
                    .partitions
                    .into_iter()
                    .map(|p| replace(p, &row_idx(), root()))
                    .collect();

                Partitioning::Partitioned(Arc::new(partitioned))
            })
            .clone()
    }
}

#[derive(Clone)]
enum Partitioning {
    // An expression that only references the row index (e.g., `row_idx == 5`).
    RowIdx(ExprRef),
    // An expression that does not reference the row index.
    Child(ExprRef),
    // Contains both the RowIdx and Child expressions, (e.g., `row_idx < child.some_field`).
    Partitioned(Arc<PartitionedExpr<Partition>>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Partition {
    RowIdx,
    Child,
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Partition::RowIdx => write!(f, "row_idx"),
            Partition::Child => write!(f, "child"),
        }
    }
}

impl LayoutReader for RowIdxLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn scope_dtype(&self) -> &ScopeDType {
        self.child.scope_dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // Since RowIdx isn't a field, we only need to register splits for the child layout
        // if there are any fields in the mask at all.
        if !field_mask.is_empty() {
            self.child.register_splits(field_mask, row_offset, splits)?;
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        match &self.partition_expr(expr) {
            Partitioning::RowIdx(expr) => Ok(Box::new(RowIdxEvaluation::new(
                self.row_offset,
                row_range,
                expr,
            ))),
            Partitioning::Child(expr) => self.child.pruning_evaluation(row_range, expr),
            Partitioning::Partitioned(..) => Ok(Box::new(NoOpPruningEvaluation)),
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        match &self.partition_expr(expr) {
            // Since this is run during pruning, we skip re-evaluating the row index expression
            // during the filter evaluation.
            Partitioning::RowIdx(_) => Ok(Box::new(NoOpMaskEvaluation)),
            Partitioning::Child(expr) => self.child.filter_evaluation(row_range, expr),
            Partitioning::Partitioned(p) => Ok(Box::new(PartitionedMaskEvaluation::try_new(
                p.clone(),
                |annotation, expr| match annotation {
                    Partition::RowIdx => Ok(Box::new(RowIdxEvaluation::new(
                        self.row_offset,
                        row_range,
                        expr,
                    ))),
                    Partition::Child => self.child.filter_evaluation(row_range, expr),
                },
                |annotation, expr| match annotation {
                    Partition::RowIdx => Ok(Box::new(RowIdxEvaluation::new(
                        self.row_offset,
                        row_range,
                        expr,
                    ))),
                    Partition::Child => self.child.projection_evaluation(row_range, expr),
                },
            )?)),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        match &self.partition_expr(expr) {
            Partitioning::RowIdx(expr) => Ok(Box::new(RowIdxEvaluation::new(
                self.row_offset,
                row_range,
                expr,
            ))),
            Partitioning::Child(expr) => self.child.projection_evaluation(row_range, expr),
            Partitioning::Partitioned(p) => Ok(Box::new(PartitionedArrayEvaluation::try_new(
                p.clone(),
                |annotation, expr| match annotation {
                    Partition::RowIdx => Ok(Box::new(RowIdxEvaluation::new(
                        self.row_offset,
                        row_range,
                        expr,
                    ))),
                    Partition::Child => self.child.projection_evaluation(row_range, expr),
                },
            )?)),
        }
    }
}

/// We need a custom RowIdx evaluation because we need to defer creating the SequenceArray until
/// we are given the final row_offset. We cannot just create a RowIdxLayout that spans the entire
/// dataset because arrays can only cover up to usize rows, not u64.
struct RowIdxEvaluation {
    array: ArrayRef,
    expr: ExprRef,
}

impl RowIdxEvaluation {
    fn new(row_offset: u64, row_range: &Range<u64>, expr: &ExprRef) -> Self {
        let array = SequenceArray::new(
            PValue::U64(row_offset + row_range.start),
            PValue::U64(1),
            PType::U64,
            usize::try_from(row_range.end - row_range.start)
                .vortex_expect("Row range length must fit in usize"),
        )
        .vortex_expect("Failed to create row index array");

        Self {
            array: array.into_array(),
            expr: expr.clone(),
        }
    }
}

#[async_trait]
impl PruningEvaluation for RowIdxEvaluation {
    async fn invoke(&self, _mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): we could optimize this if the mask was already quite sparse.
        Mask::try_from(
            self.expr
                .evaluate(&Scope::new(self.array.clone()))?
                .as_ref(),
        )
    }
}

#[async_trait]
impl MaskEvaluation for RowIdxEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): we could optimize this if the mask was already quite sparse.
        let result = Mask::try_from(
            self.expr
                .evaluate(&Scope::new(self.array.clone()))?
                .as_ref(),
        )?;

        // Note that mask evaluation requires an intersection with the input mask, whereas
        // pruning evaluation does not.
        Ok(result.bitand(&mask))
    }
}

#[async_trait]
impl ArrayEvaluation for RowIdxEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let array = filter(&self.array, &mask)?;
        self.expr.evaluate(&Scope::new(array))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use futures::stream;
    use itertools::Itertools;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{ArrayContext, ToCanonical};
    use vortex_expr::{eq, gt, lit, or, root};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_idx::{RowIdxLayoutReader, row_idx};
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutReader, LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    fn flat_expr_no_row_id() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = eq(root(), lit(3i32));
            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                &BooleanBuffer::from_iter([false, false, true, false, false]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_row_id() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = gt(row_idx(), lit(3u64));
            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                &BooleanBuffer::from_iter([false, false, false, false, true]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_or() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = or(
                eq(root(), lit(3i32)),
                or(gt(row_idx(), lit(3u64)), eq(root(), lit(1i32))),
            );

            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                vec![true, false, true, false, true],
                result.boolean_buffer().iter().collect_vec()
            );
        })
    }
}
