// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

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
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::transform::BoundPartitionedExpr;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
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
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::partitioned::BoundPartitionedExprEval;

pub struct RowIdxLayoutReader {
    name: Arc<str>,
    row_offset: u64,
    child: Arc<dyn LayoutReader>,
    partition_cache: DashMap<ExactBoundExpr, Arc<OnceLock<Partitioning>>>,
    session: VortexSession,
}

impl RowIdxLayoutReader {
    pub fn new(row_offset: u64, child: Arc<dyn LayoutReader>, session: VortexSession) -> Self {
        Self {
            name: Arc::clone(child.name()),
            row_offset,
            child,
            partition_cache: DashMap::with_hasher(Default::default()),
            session,
        }
    }

    fn partition_expr(&self, expr: &BoundExpression) -> VortexResult<Partitioning> {
        let key = ExactBoundExpr(expr.clone());

        // Check cache first with read-only lock.
        if let Some(entry) = self.partition_cache.get(&key)
            && let Some(partitioning) = entry.value().get()
        {
            return Ok(partitioning.clone());
        }

        let result = self.compute_partitioning(expr)?;

        self.partition_cache
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .get_or_init(|| result.clone());

        Ok(result)
    }

    fn compute_partitioning(&self, expr: &BoundExpression) -> VortexResult<Partitioning> {
        // Partition the expression into row idx and child expressions.
        let mut partitioned = partition_bound(expr.clone(), |expr: &BoundExpression| {
            if expr
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdx>())
            {
                vec![Partition::RowIdx]
            } else if expr.is_root() {
                vec![Partition::Child]
            } else {
                vec![]
            }
        })?;

        // If there's only a single partition, we can directly return the expression.
        if partitioned.partitions.len() == 1 {
            return Ok(match &partitioned.partition_annotations[0] {
                Partition::RowIdx => Partitioning::RowIdx(replace_row_idx(expr.clone())?),
                Partition::Child => Partitioning::Child(expr.clone()),
            });
        }

        // Replace the row_idx expression with the root expression in the row_idx partition.
        let partitions = partitioned
            .partitions
            .iter()
            .cloned()
            .map(replace_row_idx)
            .collect::<VortexResult<Vec<_>>>()?
            .into_boxed_slice();
        partitioned.replace_partitions(partitions)?;

        Ok(Partitioning::Partitioned(Arc::new(partitioned)))
    }
}

#[derive(Clone)]
enum Partitioning {
    // An expression that only references the row index (e.g., `row_idx == 5`).
    RowIdx(BoundExpression),
    // An expression that does not reference the row index.
    Child(BoundExpression),
    // Contains both the RowIdx and Child expressions, (e.g., `row_idx < child.some_field`).
    Partitioned(Arc<BoundPartitionedExpr<Partition>>),
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
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(match &self.partition_expr(expr)? {
            Partitioning::RowIdx(expr) => row_idx_mask_future(
                self.row_offset,
                row_range,
                expr.clone(),
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
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        match &self.partition_expr(expr)? {
            // Since this is run during pruning, we skip re-evaluating the row index expression
            // during the filter evaluation.
            Partitioning::RowIdx(_) => Ok(mask),
            Partitioning::Child(expr) => self.child.filter_evaluation(row_range, expr, mask),
            Partitioning::Partitioned(p) => Arc::clone(p).into_mask_future(
                mask,
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => Ok(row_idx_mask_future(
                        self.row_offset,
                        row_range,
                        expr.clone(),
                        mask,
                        self.session.clone(),
                    )),
                    Partition::Child => self.child.filter_evaluation(row_range, expr, mask),
                },
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => Ok(row_idx_array_future(
                        self.row_offset,
                        row_range,
                        expr.clone(),
                        mask,
                        self.session.clone(),
                    )),
                    Partition::Child => self.child.projection_evaluation(row_range, expr, mask),
                },
                self.session.clone(),
            ),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        match &self.partition_expr(expr)? {
            Partitioning::RowIdx(expr) => Ok(row_idx_array_future(
                self.row_offset,
                row_range,
                expr.clone(),
                mask,
                self.session.clone(),
            )),
            Partitioning::Child(expr) => self.child.projection_evaluation(row_range, expr, mask),
            Partitioning::Partitioned(p) => {
                Arc::clone(p).into_array_future(mask, |annotation, expr, mask| match annotation {
                    Partition::RowIdx => Ok(row_idx_array_future(
                        self.row_offset,
                        row_range,
                        expr.clone(),
                        mask,
                        self.session.clone(),
                    )),
                    Partition::Child => self.child.projection_evaluation(row_range, expr, mask),
                })
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn replace_row_idx(expr: BoundExpression) -> VortexResult<BoundExpression> {
    Ok(expr
        .transform_down(|node| {
            if node
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdx>())
            {
                Ok(Transformed {
                    value: BoundExpression::new_root(row_idx_dtype()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

fn row_idx_dtype() -> DType {
    DType::Primitive(PType::U64, NonNullable)
}

// Returns a SequenceArray representing the row indices for the given row range,
pub(crate) fn idx_array(row_offset: u64, row_range: &Range<u64>) -> SequenceArray {
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
    expr: BoundExpression,
    mask: MaskFuture,
    session: VortexSession,
) -> MaskFuture {
    let row_range = row_range.clone();
    MaskFuture::new(mask.len(), async move {
        let array = idx_array(row_offset, &row_range).into_array();

        let mut ctx = session.create_execution_ctx();
        let result_mask = array
            .apply_bound(&expr)?
            .null_as_false()
            .execute(&mut ctx)?;

        Ok(result_mask.bitand(&mask.await?))
    })
}

fn row_idx_array_future(
    row_offset: u64,
    row_range: &Range<u64>,
    expr: BoundExpression,
    mask: MaskFuture,
    session: VortexSession,
) -> ArrayFuture {
    let row_range = row_range.clone();
    async move {
        let array = idx_array(row_offset, &row_range).into_array();
        let filtered = array.filter(mask.await?)?;
        let mut ctx = session.create_execution_ctx();
        let array = filtered.execute::<Canonical>(&mut ctx)?.into_array();
        array.apply_bound(&expr)
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray as _;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use crate::LayoutReader;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_idx::RowIdxLayoutReader;
    use crate::layouts::row_idx::row_idx;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    #[test]
    fn flat_expr_no_row_id() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expr = eq(root(), lit(3i32));
            let reader = RowIdxLayoutReader::new(
                0,
                layout
                    .new_reader("".into(), segments, &session, &Default::default())
                    .unwrap(),
                session.clone(),
            );
            let expr = expr.bind(reader.dtype()).unwrap();
            let result = reader
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
                BoolArray::from_iter([false, false, true, false, false]),
                &mut ctx
            );
        })
    }

    #[test]
    fn flat_expr_row_id() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expr = gt(row_idx(), lit(3u64));
            let reader = RowIdxLayoutReader::new(
                0,
                layout
                    .new_reader("".into(), segments, &session, &Default::default())
                    .unwrap(),
                session.clone(),
            );
            let expr = expr.bind(reader.dtype()).unwrap();
            let result = reader
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
                BoolArray::from_iter([false, false, false, false, true]),
                &mut ctx
            );
        })
    }

    #[test]
    fn flat_expr_or() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expr = or(
                eq(root(), lit(3i32)),
                or(gt(row_idx(), lit(3u64)), eq(root(), lit(1i32))),
            );

            let reader = RowIdxLayoutReader::new(
                0,
                layout
                    .new_reader("".into(), segments, &session, &Default::default())
                    .unwrap(),
                session.clone(),
            );
            let expr = expr.bind(reader.dtype()).unwrap();
            let result = reader
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
                BoolArray::from_iter([true, false, true, false, true]),
                &mut ctx
            );
        })
    }
}
