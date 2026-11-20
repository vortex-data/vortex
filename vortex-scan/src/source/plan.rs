// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PlanDataSource`]: a [`DataSource`] over plan children, scanned by the planner-driven
//! executor.
//!
//! Nothing here is about one format. A child is a [`ScanInput`], a plan over placed segments,
//! from a Vortex file, a Parquet file whose column chunks are segments, or any store that lowers
//! its data to plans.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::ScanInput;
use super::bind_cache::BindCache;
use super::plan_pool;
use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::ScanRequest;
use crate::execution::ROW_BLOCK_SIZE;
use crate::pool::ScanPoolRef;
use crate::pool::group_contiguous;
use crate::prepare::ScanBuilder;
use crate::selection::Selection;

/// Whether a child's own statistics, such as a file footer's, prove a filter, bound to the
/// child's dtype, false for every row of the child.
pub type FilePruner = Arc<dyn Fn(&BoundExpression) -> VortexResult<bool> + Send + Sync>;

/// The row boundaries an opened scan cuts a child's work at, for the scan's projection and filter,
/// both bound to the child's dtype: ascending, covering the child's rows. A Vortex file gives its
/// layout splits over the fields the scan reads, as a V1 scan of the same fields cuts it.
pub type PlanSplitter =
    Arc<dyn Fn(&BoundExpression, Option<&BoundExpression>) -> VortexResult<Vec<u64>> + Send + Sync>;

/// A child of a [`PlanDataSource`]: its [`ScanInput`] and what the source knows of it without
/// reading payloads.
#[derive(Clone)]
pub struct PlanChild {
    input: ScanInput,
    row_count: Option<u64>,
    splitter: Option<PlanSplitter>,
    pruner: Option<FilePruner>,
}

impl PlanChild {
    /// A child scanning `input`. Without a cache, the input gets a fresh one that every scan of
    /// the child shares.
    pub fn new(mut input: ScanInput) -> Self {
        input.cache.get_or_insert_default();
        Self {
            input,
            row_count: None,
            splitter: None,
            pruner: None,
        }
    }

    /// Sets the child's exact row count. Without it, the source lowers the child's root plan to
    /// learn the count when a scan needs it.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Sets fixed row boundaries that an opened scan cuts the child's work at, whatever it reads,
    /// ascending and covering the child's rows. Without boundaries, the work is cut on the scan's
    /// chunk grid.
    pub fn with_splits(self, splits: Arc<[u64]>) -> Self {
        self.with_splitter(Arc::new(move |_, _| Ok(splits.to_vec())))
    }

    /// Sets how an opened scan finds the row boundaries to cut the child's work at, from the
    /// fields it reads. Without boundaries, the work is cut on the scan's chunk grid.
    pub fn with_splitter(mut self, splitter: PlanSplitter) -> Self {
        self.splitter = Some(splitter);
        self
    }

    /// Sets how the child's statistics rule out a scan's filter, so a scan they rule out reads
    /// nothing of the child.
    pub fn with_pruner(mut self, pruner: FilePruner) -> Self {
        self.pruner = Some(pruner);
        self
    }

    /// The input this child scans.
    pub fn input(&self) -> &ScanInput {
        &self.input
    }

    /// The child's rows, from the row count when set, else from its root plan.
    fn rows(&self) -> VortexResult<u64> {
        match self.row_count {
            Some(rows) => Ok(rows),
            None => Ok(self.input.root.get()?.row_count()),
        }
    }

    /// The bytes of all of the child's segments.
    fn byte_size(&self) -> u64 {
        self.input
            .locations
            .iter()
            .map(|location| u64::from(location.length))
            .sum()
    }

    /// The boundaries an opened scan of `projection` and `filter` cuts the child's work at, if the
    /// child has them.
    pub(super) fn splits(
        &self,
        projection: &BoundExpression,
        filter: Option<&BoundExpression>,
    ) -> VortexResult<Option<Vec<u64>>> {
        self.splitter
            .as_ref()
            .map(|splitter| splitter(projection, filter))
            .transpose()
    }

    fn is_pruned(&self, filter: Option<&BoundExpression>) -> VortexResult<bool> {
        match (&self.pruner, filter) {
            (Some(pruner), Some(filter)) => pruner(filter),
            _ => Ok(false),
        }
    }
}

/// A [`DataSource`] over plan children, scanned by the planner-driven executor.
///
/// This is the standard way to expose sources that can lower their data to plans over placed
/// segments, whatever their format. Every scan of a child shares the child's [`ScanCache`], so
/// scans that overlap in time plan it once.
///
/// A scan's request applies to each child: its row range and row selection are in the child's
/// own rows, and its `partition_range` and `partition_selection` select children by ordinal. A
/// scan reads its children in order.
///
/// [`split`](DataSourceScan::split) cuts a scan into contiguous pieces of about equal row counts,
/// each a list of children and row ranges of them, on the 8,192-row block grid that every chunk
/// boundary of the executor lies on. It reads nothing, and a scan with a limit, which holds across
/// the whole scan, is not split. [`open`](DataSourceScan::open) runs a scan on V2's planners and
/// [`Run`](crate::planning::driver::Run) driver; see `plan_pool.rs`. Partitions are the trait's
/// default, derived from the two.
///
/// [`ScanCache`]: crate::prepare::ScanCache
#[derive(Clone)]
pub struct PlanDataSource {
    session: VortexSession,
    dtype: DType,
    children: Arc<[PlanChild]>,
    row_count: Precision<u64>,
    byte_size: Precision<u64>,
}

impl PlanDataSource {
    /// A source scanning one `input` with `session`.
    pub fn new(session: VortexSession, input: ScanInput) -> Self {
        Self::from_child(session, PlanChild::new(input))
    }

    /// A source scanning one `child` with `session`.
    pub fn from_child(session: VortexSession, child: PlanChild) -> Self {
        let dtype = child.input.dtype.clone();
        Self::from_children(session, dtype, vec![child])
    }

    /// A source scanning `children` in order with `session`. Every child must have `dtype`.
    pub fn from_children(session: VortexSession, dtype: DType, children: Vec<PlanChild>) -> Self {
        let row_count = children
            .iter()
            .map(|child| child.row_count)
            .sum::<Option<u64>>()
            .map_or(Precision::Absent, Precision::exact);
        let byte_size = Precision::inexact(children.iter().map(PlanChild::byte_size).sum());
        Self {
            session,
            dtype,
            children: children.into(),
            row_count,
            byte_size,
        }
    }

    /// Sets the exact row count of a one-child source, which scans report.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        if let [child] = &*self.children {
            self.children = Arc::new([child.clone().with_row_count(row_count)]);
        }
        self.row_count = Precision::exact(row_count);
        self
    }

    /// Sets an estimate of the source's byte size.
    pub fn with_byte_size(mut self, byte_size: Precision<u64>) -> Self {
        self.byte_size = byte_size;
        self
    }

    /// The children this source scans, in order.
    pub fn children(&self) -> &[PlanChild] {
        &self.children
    }

    /// The session scans of this source run with.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

#[async_trait]
impl DataSource for PlanDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn byte_size(&self) -> Precision<u64> {
        self.byte_size
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let projection = BindCache::bind(&self.session, &request.projection, &self.dtype)?;
        let filter = request
            .filter
            .map(|filter| BindCache::bind(&self.session, &filter, &self.dtype))
            .transpose()?;
        let mut units = Vec::new();
        let scans_rows =
            request.limit != Some(0) && !request.row_range.as_ref().is_some_and(Range::is_empty);
        for (ordinal, child) in self.children.iter().enumerate().filter(|_| scans_rows) {
            let index = ordinal as u64;
            let selected = request
                .partition_range
                .as_ref()
                .is_none_or(|range| range.contains(&index))
                && request
                    .partition_selection
                    .row_count_in_range(&(index..index + 1))
                    == 1;
            if !selected || child.is_pruned(filter.as_ref())? {
                continue;
            }
            let rows = child
                .row_count
                .map(|count| clamp_rows(request.row_range.clone(), count));
            if rows.as_ref().is_some_and(Range::is_empty) {
                continue;
            }
            units.push(Unit {
                child: ordinal,
                rows,
            });
        }
        let state = ScanState {
            session: self.session.clone(),
            children: Arc::clone(&self.children),
            dtype: projection.dtype().clone(),
            projection,
            filter,
            row_range: request.row_range,
            selection: request.selection,
            ordered: request.ordered,
            limit: request.limit,
        };
        Ok(Box::new(PlanScan {
            state: Arc::new(state),
            units,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// The request's `rows`, or all of them, clamped to a child of `count` rows.
fn clamp_rows(rows: Option<Range<u64>>, count: u64) -> Range<u64> {
    let rows = rows.unwrap_or(0..count);
    let end = rows.end.min(count);
    rows.start.min(end)..end
}

/// What every piece and pool of one scan of a [`PlanDataSource`] shares.
pub(super) struct ScanState {
    pub(super) session: VortexSession,
    children: Arc<[PlanChild]>,
    /// The dtype of the scan's output.
    pub(super) dtype: DType,
    pub(super) projection: BoundExpression,
    pub(super) filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    pub(super) ordered: bool,
    pub(super) limit: Option<u64>,
}

impl ScanState {
    /// The source's child at `ordinal`.
    pub(super) fn child(&self, ordinal: usize) -> &PlanChild {
        &self.children[ordinal]
    }

    /// The rows of `unit` this scan reads: its own, or else the request's row range clamped to
    /// its child.
    pub(super) fn unit_rows(&self, unit: &Unit) -> VortexResult<Range<u64>> {
        match &unit.rows {
            Some(rows) => Ok(rows.clone()),
            None => Ok(clamp_rows(
                self.row_range.clone(),
                self.child(unit.child).rows()?,
            )),
        }
    }

    /// A builder of this scan's `rows` of `child`.
    pub(super) fn builder(&self, child: &PlanChild, rows: Range<u64>) -> ScanBuilder {
        ScanBuilder::new(self.session.clone(), child.input.clone())
            .with_projection(self.projection.clone())
            .with_some_filter(self.filter.clone())
            .with_selection(self.selection.clone())
            .with_row_range(rows)
    }
}

/// Rows of one child that a scan reads.
#[derive(Clone, Debug)]
pub(super) struct Unit {
    /// The child's ordinal in the source.
    pub(super) child: usize,
    /// The child's rows, or `None` for the request's rows of a child whose row count is not known
    /// until its plan is lowered.
    pub(super) rows: Option<Range<u64>>,
}

/// A scan of a [`PlanDataSource`], or a piece split from one.
struct PlanScan {
    state: Arc<ScanState>,
    /// The scan's units, in child and row order.
    units: Vec<Unit>,
}

impl PlanScan {
    /// The rows of `unit` the scan selects before filtering, or `None` when not known yet.
    fn unit_rows(&self, unit: &Unit) -> Option<u64> {
        let rows = unit.rows.as_ref()?;
        Some(self.state.selection.row_count_in_range(rows))
    }

    /// The bytes of `unit`, in proportion to its share of its child's rows.
    fn unit_bytes(&self, unit: &Unit) -> u64 {
        let child = self.state.child(unit.child);
        let bytes = child.byte_size();
        match (&unit.rows, child.row_count) {
            (Some(rows), Some(count)) if count > 0 => u64::try_from(
                u128::from(bytes) * u128::from(rows.end - rows.start) / u128::from(count),
            )
            .unwrap_or(u64::MAX),
            _ => bytes,
        }
    }
}

impl DataSourceScan for PlanScan {
    fn dtype(&self) -> &DType {
        &self.state.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        let Some(rows) = self
            .units
            .iter()
            .map(|unit| self.unit_rows(unit))
            .sum::<Option<u64>>()
        else {
            return Precision::Absent;
        };
        let rows = self.state.limit.map_or(rows, |limit| rows.min(limit));
        if self.state.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::inexact(self.units.iter().map(|unit| self.unit_bytes(unit)).sum())
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        // A limit holds across the whole scan, which pieces read independently cannot share.
        if self.state.limit.is_some() || n.get() == 1 {
            return Ok(vec![self]);
        }
        let atoms = split_atoms(&self.units, n);
        if atoms.len() <= 1 {
            return Ok(vec![self]);
        }
        let known: Vec<u64> = atoms
            .iter()
            .filter_map(|unit| unit.rows.as_ref().map(|rows| rows.end - rows.start))
            .collect();
        // A deferred child weighs as much as the average ready unit.
        let unknown = (known.iter().sum::<u64>() / known.len().max(1) as u64).max(1);
        let weight = |unit: &Unit| {
            unit.rows
                .as_ref()
                .map_or(unknown, |rows| rows.end - rows.start)
        };
        let state = self.state;
        Ok(group_contiguous(atoms, n, weight)
            .into_iter()
            .map(|atoms| {
                Box::new(PlanScan {
                    state: Arc::clone(&state),
                    units: join_atoms(atoms),
                }) as DataSourceScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        plan_pool::open(self.state, self.units)
    }
}

/// The pieces a split may not cut further: the scan's known rows on a grid of whole 8,192-row
/// blocks, coarse enough that each of `n` pieces gets about 64 of them, and each unit whose rows
/// are not known yet whole.
fn split_atoms(units: &[Unit], n: NonZeroUsize) -> Vec<Unit> {
    let block = ROW_BLOCK_SIZE as u64;
    let known: u64 = units
        .iter()
        .filter_map(|unit| unit.rows.as_ref().map(|rows| rows.end - rows.start))
        .sum();
    let per_atom = known.div_ceil(n.get() as u64 * 64).div_ceil(block).max(1) * block;
    let mut atoms = Vec::new();
    for unit in units {
        let Some(rows) = &unit.rows else {
            atoms.push(unit.clone());
            continue;
        };
        let mut start = rows.start;
        while start < rows.end {
            let end = ((start / per_atom + 1) * per_atom).min(rows.end);
            atoms.push(Unit {
                child: unit.child,
                rows: Some(start..end),
            });
            start = end;
        }
    }
    atoms
}

/// Joins adjacent atoms of the same child back into one unit.
fn join_atoms(atoms: Vec<Unit>) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    for atom in atoms {
        if let Some(last) = units.last_mut()
            && last.child == atom.child
            && let (Some(rows), Some(next)) = (&mut last.rows, &atom.rows)
            && rows.end == next.start
        {
            rows.end = next.end;
            continue;
        }
        units.push(atom);
    }
    units
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;

    use super::Unit;
    use super::join_atoms;
    use super::split_atoms;

    fn unit(child: usize, rows: std::ops::Range<u64>) -> Unit {
        Unit {
            child,
            rows: Some(rows),
        }
    }

    #[rstest]
    #[case::one_block(vec![unit(0, 0..100)], 4, 1)]
    #[case::grid_aligned(vec![unit(0, 10..40_000)], 1, 5)]
    fn atoms_lie_on_the_block_grid(
        #[case] units: Vec<Unit>,
        #[case] n: usize,
        #[case] expected: usize,
    ) {
        let n = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        let atoms = split_atoms(&units, n);
        assert_eq!(atoms.len(), expected);
        for atom in &atoms[1..] {
            let start = atom.rows.as_ref().map_or(0, |rows| rows.start);
            assert_eq!(start % 8192, 0);
        }
        let joined = join_atoms(atoms);
        assert_eq!(
            joined.iter().map(|u| u.rows.clone()).collect::<Vec<_>>(),
            units.iter().map(|u| u.rows.clone()).collect::<Vec<_>>()
        );
    }
}
