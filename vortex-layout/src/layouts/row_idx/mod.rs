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
use vortex_array::Canonical;
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
use vortex_array::expr::and_collect;
use vortex_array::expr::forms::conjuncts;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;
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
            name: Arc::clone(child.name()),
            row_offset,
            child,
            partition_cache: DashMap::with_hasher(Default::default()),
            session,
        }
    }

    fn partition_expr(&self, expr: &Expression) -> VortexResult<Partitioning> {
        let key = ExactExpr(expr.clone());

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

    fn compute_partitioning(&self, expr: &Expression) -> VortexResult<Partitioning> {
        // Partition the expression into row idx and child expressions.
        let mut partitioned = partition(expr.clone(), self.dtype(), |expr| {
            if expr.is::<RowIdx>() {
                vec![Partition::RowIdx]
            } else if is_root(expr) {
                vec![Partition::Child]
            } else {
                vec![]
            }
        })?;

        // If there's only a single partition, we can directly return the expression.
        if partitioned.partitions.len() == 1 {
            return Ok(match &partitioned.partition_annotations[0] {
                Partition::RowIdx => {
                    Partitioning::RowIdx(replace(expr.clone(), &row_idx(), root()))
                }
                Partition::Child => Partitioning::Child(expr.clone()),
            });
        }

        // Replace the row_idx expression with the root expression in the row_idx partition.
        partitioned.partitions = partitioned
            .partitions
            .into_iter()
            .map(|p| replace(p, &row_idx(), root()))
            .collect();

        Ok(Partitioning::Partitioned(Arc::new(partitioned)))
    }
}

pub(crate) struct ExtractedRowIdxFilter {
    pub(crate) filter: Option<Expression>,
    pub(crate) selection: Selection,
    pub(crate) row_range: Option<Range<u64>>,
}

pub(crate) fn extract_row_idx_filter(
    filter: &Expression,
    row_offset: u64,
    row_count: u64,
) -> ExtractedRowIdxFilter {
    let mut remaining = Vec::new();
    let mut selection = Selection::All;
    let mut row_range = None;

    for conjunct in conjuncts(filter) {
        match extract_row_idx_conjunct(&conjunct, row_offset, row_count) {
            Some(RowIdxFilterPart::Selection(indices)) => {
                intersect_selection(&mut selection, indices);
            }
            Some(RowIdxFilterPart::Range(range)) => {
                intersect_row_range(&mut row_range, range);
            }
            None => remaining.push(conjunct),
        }
    }

    normalize_selection_and_range(&mut selection, &mut row_range);

    ExtractedRowIdxFilter {
        filter: and_collect(remaining),
        selection,
        row_range,
    }
}

enum RowIdxFilterPart {
    Selection(Buffer<u64>),
    Range(Range<u64>),
}

fn extract_row_idx_conjunct(
    expr: &Expression,
    row_offset: u64,
    row_count: u64,
) -> Option<RowIdxFilterPart> {
    extract_row_idx_binary(expr, row_offset, row_count)
        .or_else(|| extract_row_idx_in_list(expr, row_offset, row_count))
}

fn extract_row_idx_binary(
    expr: &Expression,
    row_offset: u64,
    row_count: u64,
) -> Option<RowIdxFilterPart> {
    let operator = *expr.as_opt::<Binary>()?;
    let (operator, scalar) = if expr.child(0).is::<RowIdx>() {
        (operator, expr.child(1).as_opt::<Literal>()?)
    } else if expr.child(1).is::<RowIdx>() {
        (swap_operator(operator)?, expr.child(0).as_opt::<Literal>()?)
    } else {
        return None;
    };

    let Some(value) = literal_to_u64(scalar)? else {
        return Some(RowIdxFilterPart::Selection(empty_indices()));
    };

    match operator {
        Operator::Eq => Some(RowIdxFilterPart::Selection(Buffer::from_iter(
            relative_index(value, row_offset, row_count),
        ))),
        Operator::Gt => Some(RowIdxFilterPart::Range(relative_range(
            value.saturating_add(1)..u64::MAX,
            row_offset,
            row_count,
        ))),
        Operator::Gte => Some(RowIdxFilterPart::Range(relative_range(
            value..u64::MAX,
            row_offset,
            row_count,
        ))),
        Operator::Lt => Some(RowIdxFilterPart::Range(relative_range(
            0..value,
            row_offset,
            row_count,
        ))),
        Operator::Lte => Some(RowIdxFilterPart::Range(relative_range(
            0..value.saturating_add(1),
            row_offset,
            row_count,
        ))),
        _ => None,
    }
}

fn extract_row_idx_in_list(
    expr: &Expression,
    row_offset: u64,
    row_count: u64,
) -> Option<RowIdxFilterPart> {
    expr.as_opt::<ListContains>()?;

    if !expr.child(1).is::<RowIdx>() {
        return None;
    }

    let list = expr.child(0).as_opt::<Literal>()?.as_list_opt()?;
    let mut indices = Vec::new();
    for scalar in list.elements()? {
        let Some(value) = literal_to_u64(&scalar)? else {
            continue;
        };
        indices.extend(relative_index(value, row_offset, row_count));
    }
    indices.sort_unstable();
    indices.dedup();

    Some(RowIdxFilterPart::Selection(Buffer::from_iter(indices)))
}

fn swap_operator(operator: Operator) -> Option<Operator> {
    Some(match operator {
        Operator::Eq => Operator::Eq,
        Operator::Gt => Operator::Lt,
        Operator::Gte => Operator::Lte,
        Operator::Lt => Operator::Gt,
        Operator::Lte => Operator::Gte,
        _ => return None,
    })
}

fn literal_to_u64(scalar: &Scalar) -> Option<Option<u64>> {
    scalar.as_primitive_opt()?.as_opt::<u64>()
}

fn relative_index(value: u64, row_offset: u64, row_count: u64) -> Option<u64> {
    let row_end = row_offset.saturating_add(row_count);
    (row_offset..row_end)
        .contains(&value)
        .then(|| value - row_offset)
}

fn relative_range(range: Range<u64>, row_offset: u64, row_count: u64) -> Range<u64> {
    let row_end = row_offset.saturating_add(row_count);
    let start = range.start.max(row_offset);
    let end = range.end.min(row_end);

    if start >= end {
        0..0
    } else {
        start - row_offset..end - row_offset
    }
}

fn empty_indices() -> Buffer<u64> {
    Buffer::from_iter(std::iter::empty::<u64>())
}

fn intersect_selection(selection: &mut Selection, indices: Buffer<u64>) {
    match selection {
        Selection::All => {
            *selection = Selection::IncludeByIndex(indices);
        }
        Selection::IncludeByIndex(existing) => {
            *selection = Selection::IncludeByIndex(Buffer::from_iter(intersect_sorted(
                existing.as_slice(),
                indices.as_slice(),
            )));
        }
        Selection::ExcludeByIndex(_)
        | Selection::IncludeRoaring(_)
        | Selection::ExcludeRoaring(_) => {}
    }
}

fn intersect_sorted(left: &[u64], right: &[u64]) -> Vec<u64> {
    let mut result = Vec::new();
    let (mut left_idx, mut right_idx) = (0, 0);
    while left_idx < left.len() && right_idx < right.len() {
        match left[left_idx].cmp(&right[right_idx]) {
            std::cmp::Ordering::Equal => {
                result.push(left[left_idx]);
                left_idx += 1;
                right_idx += 1;
            }
            std::cmp::Ordering::Less => left_idx += 1,
            std::cmp::Ordering::Greater => right_idx += 1,
        }
    }
    result
}

fn intersect_row_range(row_range: &mut Option<Range<u64>>, next: Range<u64>) {
    *row_range = Some(match row_range.take() {
        Some(existing) => existing.start.max(next.start)..existing.end.min(next.end),
        None => next,
    });
}

fn normalize_selection_and_range(selection: &mut Selection, row_range: &mut Option<Range<u64>>) {
    if row_range.as_ref().is_some_and(|range| range.is_empty()) {
        *selection = Selection::IncludeByIndex(empty_indices());
        *row_range = None;
        return;
    }

    if !matches!(selection, Selection::IncludeByIndex(_)) {
        return;
    }

    let Some(range) = row_range.take() else {
        return;
    };

    let Selection::IncludeByIndex(indices) = selection else {
        unreachable!("row range only removed for include-by-index selection");
    };
    *indices = Buffer::from_iter(indices.iter().copied().filter(|idx| range.contains(idx)));
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
        Ok(match &self.partition_expr(expr)? {
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
                        expr,
                        mask,
                        self.session.clone(),
                    )),
                    Partition::Child => self.child.filter_evaluation(row_range, expr, mask),
                },
                |annotation, expr, mask| match annotation {
                    Partition::RowIdx => Ok(row_idx_array_future(
                        self.row_offset,
                        row_range,
                        expr,
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
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        match &self.partition_expr(expr)? {
            Partitioning::RowIdx(expr) => Ok(row_idx_array_future(
                self.row_offset,
                row_range,
                expr,
                mask,
                self.session.clone(),
            )),
            Partitioning::Child(expr) => self.child.projection_evaluation(row_range, expr, mask),
            Partitioning::Partitioned(p) => {
                Arc::clone(p).into_array_future(mask, |annotation, expr, mask| match annotation {
                    Partition::RowIdx => Ok(row_idx_array_future(
                        self.row_offset,
                        row_range,
                        expr,
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
    session: VortexSession,
) -> ArrayFuture {
    let row_range = row_range.clone();
    let expr = expr.clone();
    async move {
        let array = idx_array(row_offset, &row_range).into_array();
        let filtered = array.filter(mask.await?)?;
        let mut ctx = session.create_execution_ctx();
        let array = filtered.execute::<Canonical>(&mut ctx)?.into_array();
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
    use vortex_io::session::RuntimeSessionExt;

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
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
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
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
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
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = buffer![1..=5].into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
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
