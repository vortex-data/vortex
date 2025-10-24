// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use Nullability::NonNullable;
pub use expr::*;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::compute::filter;
use vortex_array::stats::Precision;
use vortex_array::{ArrayRef, IntoArray, MaskFuture};
use vortex_dtype::{DType, FieldMask, FieldName, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::{PartitionedExpr, partition, replace};
use vortex_expr::{ExactExpr, ExprRef, Scope, is_root, root};
use vortex_mask::Mask;
use vortex_scalar::PValue;
use vortex_sequence::SequenceArray;
use vortex_utils::aliases::dash_map::DashMap;

use crate::layouts::partitioned::PartitionedExprEval;
use crate::{ArrayFuture, LayoutReader};

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
            partition_cache: DashMap::with_hasher(Default::default()),
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

impl Partition {
    pub fn name(&self) -> &str {
        match self {
            Partition::RowIdx => "row_idx",
            Partition::Child => "child",
        }
    }
}

impl From<Partition> for FieldName {
    fn from(value: Partition) -> Self {
        FieldName::from(value.name())
    }
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl LayoutReader for RowIdxLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
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
        self.child.register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(match &self.partition_expr(expr) {
            Partitioning::RowIdx(expr) => {
                row_idx_mask_future(self.row_offset, row_range, expr, MaskFuture::ready(mask))
            }
            Partitioning::Child(expr) => self.child.pruning_evaluation(row_range, expr, mask)?,
            Partitioning::Partitioned(..) => MaskFuture::ready(mask),
        })
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        match &self.partition_expr(expr) {
            // Since this is run during pruning, we skip re-evaluating the row index expression
            // during the filter evaluation.
            Partitioning::RowIdx(_) => Ok(mask),
            Partitioning::Child(expr) => self.child.filter_evaluation(row_range, expr, mask),
            Partitioning::Partitioned(p) => p.clone().into_mask_future(
                mask,
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => {
                        Ok(row_idx_mask_future(self.row_offset, row_range, expr, mask))
                    }
                    Partition::Child => self.child.filter_evaluation(row_range, expr, mask),
                },
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => {
                        Ok(row_idx_array_future(self.row_offset, row_range, expr, mask))
                    }
                    Partition::Child => self.child.projection_evaluation(row_range, expr, mask),
                },
            ),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        match &self.partition_expr(expr) {
            Partitioning::RowIdx(expr) => {
                Ok(row_idx_array_future(self.row_offset, row_range, expr, mask))
            }
            Partitioning::Child(expr) => self.child.projection_evaluation(row_range, expr, mask),
            Partitioning::Partitioned(p) => {
                p.clone()
                    .into_array_future(mask, |annotation, expr, mask| match annotation {
                        Partition::RowIdx => {
                            Ok(row_idx_array_future(self.row_offset, row_range, expr, mask))
                        }
                        Partition::Child => self.child.projection_evaluation(row_range, expr, mask),
                    })
            }
        }
    }
}

// Returns a SequenceArray representing the row indices for the given row range,
fn idx_array(row_offset: u64, row_range: &Range<u64>) -> SequenceArray {
    SequenceArray::new(
        PValue::U64(row_offset + row_range.start),
        PValue::U64(1),
        PType::U64,
        NonNullable,
        usize::try_from(row_range.end - row_range.start)
            .vortex_expect("Row range length must fit in usize"),
    )
    .vortex_expect("Failed to create row index array")
}

fn row_idx_mask_future(
    row_offset: u64,
    row_range: &Range<u64>,
    expr: &ExprRef,
    mask: MaskFuture,
) -> MaskFuture {
    let row_range = row_range.clone();
    let expr = expr.clone();
    MaskFuture::new(mask.len(), async move {
        let array = idx_array(row_offset, &row_range).into_array();
        let result_mask = expr
            .evaluate(&Scope::new(array))?
            .try_to_mask_fill_null_false()?;
        Ok(result_mask.bitand(&mask.await?))
    })
}

fn row_idx_array_future(
    row_offset: u64,
    row_range: &Range<u64>,
    expr: &ExprRef,
    mask: MaskFuture,
) -> ArrayFuture {
    let row_range = row_range.clone();
    let expr = expr.clone();
    async move {
        let array = idx_array(row_offset, &row_range).into_array();
        let array = filter(&array, &mask.await?)?;
        expr.evaluate(&Scope::new(array))
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use vortex_array::{ArrayContext, IntoArray as _, MaskFuture, ToCanonical};
    use vortex_buffer::{BitBuffer, buffer};
    use vortex_expr::{eq, gt, lit, or, root};
    use vortex_io::runtime::single::block_on;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_idx::{RowIdxLayoutReader, row_idx};
    use crate::segments::TestSegments;
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};
    use crate::{LayoutReader, LayoutStrategy};

    #[test]
    fn flat_expr_no_row_id() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let expr = eq(root(), lit(3i32));
            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments).unwrap())
                    .projection_evaluation(
                        &(0..layout.row_count()),
                        &expr,
                        MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                    )
                    .unwrap()
                    .await
                    .unwrap()
                    .to_bool();

            assert_eq!(
                &BitBuffer::from_iter([false, false, true, false, false]),
                result.bit_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_row_id() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let expr = gt(row_idx(), lit(3u64));
            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments).unwrap())
                    .projection_evaluation(
                        &(0..layout.row_count()),
                        &expr,
                        MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                    )
                    .unwrap()
                    .await
                    .unwrap()
                    .to_bool();

            assert_eq!(
                &BitBuffer::from_iter([false, false, false, false, true]),
                result.bit_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_or() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let expr = or(
                eq(root(), lit(3i32)),
                or(gt(row_idx(), lit(3u64)), eq(root(), lit(1i32))),
            );

            let result =
                RowIdxLayoutReader::new(0, layout.new_reader("".into(), segments).unwrap())
                    .projection_evaluation(
                        &(0..layout.row_count()),
                        &expr,
                        MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                    )
                    .unwrap()
                    .await
                    .unwrap()
                    .to_bool();

            assert_eq!(
                vec![true, false, true, false, true],
                result.bit_buffer().iter().collect_vec()
            );
        })
    }
}
