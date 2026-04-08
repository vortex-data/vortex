// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

use std::collections::BTreeSet;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use Nullability::NonNullable;
pub use expr::*;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::ExactExpr;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::scalar::PValue;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_sequence::Sequence;
use vortex_sequence::SequenceArray;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::layouts::partitioned::PartitionedExprEval;

pub struct RowIdxLayoutReader {
    name: Arc<str>,
    row_offset: u64,
    child: Arc<dyn LayoutReader>,
    partition_cache: DashMap<ExactExpr, Arc<OnceLock<Partitioning>>>,
    session: VortexSession,
}

impl RowIdxLayoutReader {
    pub fn new(row_offset: u64, child: Arc<dyn LayoutReader>, session: VortexSession) -> Self {
        Self {
            name: child.name().clone(),
            row_offset,
            child,
            partition_cache: DashMap::with_hasher(Default::default()),
            session,
        }
    }

    fn partition_expr(&self, expr: &Expression) -> Partitioning {
        let key = ExactExpr(expr.clone());

        // Check cache first with read-only lock.
        if let Some(entry) = self.partition_cache.get(&key)
            && let Some(partitioning) = entry.value().get()
        {
            return partitioning.clone();
        }

        let cell = self
            .partition_cache
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone();

        cell.get_or_init(|| self.compute_partitioning(expr)).clone()
    }

    fn compute_partitioning(&self, expr: &Expression) -> Partitioning {
        // Partition the expression into row idx and child expressions.
        let mut partitioned = partition(expr.clone(), self.dtype(), |expr| {
            if expr.is::<RowIdx>() {
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
    }
}

#[derive(Clone)]
enum Partitioning {
    // An expression that only references the row index (e.g., `row_idx == 5`).
    RowIdx(Expression),
    // An expression that does not reference the row index.
    Child(Expression),
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

    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(match &self.partition_expr(expr) {
            Partitioning::RowIdx(expr) => row_idx_mask_future(
                self.row_offset,
                row_range,
                expr,
                MaskFuture::ready(mask),
                self.session.clone(),
            ),
            Partitioning::Child(expr) => self.child.pruning_evaluation(row_range, expr, mask)?,
            Partitioning::Partitioned(..) => MaskFuture::ready(mask),
        })
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
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
                    Partition::RowIdx => Ok(row_idx_mask_future(
                        self.row_offset,
                        row_range,
                        expr,
                        mask,
                        self.session.clone(),
                    )),
                    Partition::Child => self.child.filter_evaluation(row_range, expr, mask),
                },
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => {
                        Ok(row_idx_array_future(self.row_offset, row_range, expr, mask))
                    }
                    Partition::Child => self.child.projection_evaluation(row_range, expr, mask),
                },
                self.session.clone(),
            ),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
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
    Sequence::try_new(
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
    expr: &Expression,
    mask: MaskFuture,
    session: VortexSession,
) -> MaskFuture {
    let row_range = row_range.clone();
    let expr = expr.clone();
    MaskFuture::new(mask.len(), async move {
        let array = idx_array(row_offset, &row_range).into_array();

        let mut ctx = session.create_execution_ctx();
        let result_mask = array.apply(&expr)?.execute::<Mask>(&mut ctx)?;

        Ok(result_mask.bitand(&mask.await?))
    })
}

fn row_idx_array_future(
    row_offset: u64,
    row_range: &Range<u64>,
    expr: &Expression,
    mask: MaskFuture,
) -> ArrayFuture {
    let row_range = row_range.clone();
    let expr = expr.clone();
    async move {
        let array = idx_array(row_offset, &row_range).into_array();
        let array = array.filter(mask.await?)?.to_canonical()?.into_array();
        array.apply(&expr)
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray as _;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;

    use crate::LayoutReader;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_idx::RowIdxLayoutReader;
    use crate::layouts::row_idx::row_idx;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

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
            let result = RowIdxLayoutReader::new(
                0,
                layout.new_reader("".into(), segments, &SESSION).unwrap(),
                SESSION.clone(),
            )
            .projection_evaluation(
                &(0..layout.row_count()),
                &expr,
                MaskFuture::new_true(layout.row_count().try_into().unwrap()),
            )
            .unwrap()
            .await
            .unwrap();

            assert_arrays_eq!(
                result,
                BoolArray::from_iter([false, false, true, false, false])
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
            let result = RowIdxLayoutReader::new(
                0,
                layout.new_reader("".into(), segments, &SESSION).unwrap(),
                SESSION.clone(),
            )
            .projection_evaluation(
                &(0..layout.row_count()),
                &expr,
                MaskFuture::new_true(layout.row_count().try_into().unwrap()),
            )
            .unwrap()
            .await
            .unwrap();

            assert_arrays_eq!(
                result,
                BoolArray::from_iter([false, false, false, false, true])
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

            let result = RowIdxLayoutReader::new(
                0,
                layout.new_reader("".into(), segments, &SESSION).unwrap(),
                SESSION.clone(),
            )
            .projection_evaluation(
                &(0..layout.row_count()),
                &expr,
                MaskFuture::new_true(layout.row_count().try_into().unwrap()),
            )
            .unwrap()
            .await
            .unwrap();

            assert_arrays_eq!(
                result,
                BoolArray::from_iter([true, false, true, false, true])
            );
        })
    }
}
