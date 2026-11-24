// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PlanDataSource`]: a [`DataSource`] over a plan child, scanned by the planner-driven
//! executor.
//!
//! Nothing here is about one format. A child is a [`ScanInput`], a plan over placed segments,
//! from a Vortex file, a Parquet file whose column chunks are segments, or any store that lowers
//! its data to plans.

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::ScanInput;
use super::plan_stream;
use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::prepare::ScanBuilder;
use crate::selection::Selection;
use crate::stream::ScanExec;

/// Whether a child's own statistics, such as a file footer's, prove a filter, bound to the
/// child's dtype, false for every row of the child.
pub type FilePruner = Arc<dyn Fn(&BoundExpression) -> VortexResult<bool> + Send + Sync>;

/// The row boundaries an opened scan cuts a child's work at, for the scan's projection and filter,
/// both bound to the child's dtype: ascending, covering the child's rows. A Vortex file gives its
/// layout splits over the fields the scan reads, as a V1 scan of the same fields cuts it.
pub type PlanSplitter =
    Arc<dyn Fn(&BoundExpression, Option<&BoundExpression>) -> VortexResult<Vec<u64>> + Send + Sync>;

/// The child of a [`PlanDataSource`]: its [`ScanInput`] and what the source knows of it without
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

/// A [`DataSource`] over one plan child, scanned by the planner-driven executor.
///
/// This is the standard way to expose sources that can lower their data to plans over placed
/// segments, whatever their format. Every scan shares the child's [`ScanCache`], so scans that
/// overlap in time plan it once.
///
/// A scan's row range and row selection are in the child's rows. Its `partition_range` and
/// `partition_selection` select the child as partition 0.
///
/// Scans are unordered: a request for an ordered scan is an error. Making a scan does the reads it
/// needs to know its units of work: it prepares the child and cuts it into parts, which for a
/// filter reads zone maps to drop the parts they rule out. [`split`](DataSourceScan::split) and
/// [`open`](DataSourceScan::open) read nothing. [`split`](DataSourceScan::split) returns a scan
/// whole. [`open`](DataSourceScan::open) runs it on V2's planners and
/// [`Run`](crate::planning::driver::Run) driver; see `plan_stream.rs`. A scan's one partition reads
/// its opened stream.
///
/// [`ScanCache`]: crate::prepare::ScanCache
#[derive(Clone)]
pub struct PlanDataSource {
    session: VortexSession,
    child: PlanChild,
    row_count: Precision<u64>,
    byte_size: Precision<u64>,
}

impl PlanDataSource {
    /// A source scanning `input` with `session`.
    pub fn new(session: VortexSession, input: ScanInput) -> Self {
        Self::from_child(session, PlanChild::new(input))
    }

    /// A source scanning `child` with `session`.
    pub fn from_child(session: VortexSession, child: PlanChild) -> Self {
        let row_count = child.row_count.map_or(Precision::Absent, Precision::exact);
        let byte_size = Precision::inexact(child.byte_size());
        Self {
            session,
            child,
            row_count,
            byte_size,
        }
    }

    /// Sets the child's exact row count, which scans report.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.child = self.child.with_row_count(row_count);
        self.row_count = Precision::exact(row_count);
        self
    }
}

#[async_trait]
impl DataSource for PlanDataSource {
    fn dtype(&self) -> &DType {
        &self.child.input.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn byte_size(&self) -> Precision<u64> {
        self.byte_size
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        if request.ordered {
            vortex_bail!("PlanDataSource does not support ordered scans");
        }
        let dtype = self.dtype();
        let projection = request.projection.bind(dtype)?.optimize_recursive()?;
        let filter = request
            .filter
            .map(|filter| filter.bind(dtype)?.optimize_recursive())
            .transpose()?;
        let selected = request
            .partition_range
            .as_ref()
            .is_none_or(|range| range.contains(&0))
            && request.partition_selection.row_count_in_range(&(0..1)) == 1;
        let scans_rows =
            request.limit != Some(0) && !request.row_range.as_ref().is_some_and(Range::is_empty);
        let unit = if selected && scans_rows && !self.child.is_pruned(filter.as_ref())? {
            let rows = self
                .child
                .row_count
                .map(|count| clamp_rows(request.row_range.clone(), count));
            (!rows.as_ref().is_some_and(Range::is_empty)).then_some(Unit { rows })
        } else {
            None
        };
        let state = ScanState {
            session: self.session.clone(),
            child: self.child.clone(),
            dtype: projection.dtype().clone(),
            projection,
            filter,
            row_range: request.row_range,
            selection: request.selection,
            limit: request.limit,
        };
        // The scan's parts are its units of work, and cutting them can read zone maps, so it
        // happens here, where the host awaits the scan, and opening reads nothing.
        let planned = plan_stream::plan(&state, unit.as_ref()).await?;
        Ok(Box::new(PlanScan {
            state: Arc::new(state),
            planned,
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

/// What every stream of one scan of a [`PlanDataSource`] shares.
pub(super) struct ScanState {
    pub(super) session: VortexSession,
    pub(super) child: PlanChild,
    /// The dtype of the scan's output.
    pub(super) dtype: DType,
    pub(super) projection: BoundExpression,
    pub(super) filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    pub(super) limit: Option<u64>,
}

impl ScanState {
    /// The rows of `unit` this scan reads: its own, or else the request's row range clamped to
    /// the child.
    pub(super) fn unit_rows(&self, unit: &Unit) -> VortexResult<Range<u64>> {
        match &unit.rows {
            Some(rows) => Ok(rows.clone()),
            None => Ok(clamp_rows(self.row_range.clone(), self.child.rows()?)),
        }
    }

    /// A builder of this scan's `rows` of the child.
    pub(super) fn builder(&self, rows: Range<u64>) -> ScanBuilder {
        ScanBuilder::new(self.session.clone(), self.child.input.clone())
            .with_projection(self.projection.clone())
            .with_some_filter(self.filter.clone())
            .with_selection(self.selection.clone())
            .with_row_range(rows)
    }
}

/// The rows of the child that a scan reads.
#[derive(Clone, Debug)]
pub(super) struct Unit {
    /// The child's rows, or `None` for the request's rows of a child whose row count is not known
    /// until its plan is lowered.
    pub(super) rows: Option<Range<u64>>,
}

/// A scan of a [`PlanDataSource`].
struct PlanScan {
    state: Arc<ScanState>,
    /// The scan's parts, cut when the scan was made.
    planned: plan_stream::Planned,
}

impl DataSourceScan for PlanScan {
    fn dtype(&self) -> &DType {
        &self.state.dtype
    }

    fn split(self: Box<Self>, _n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        Ok(vec![self])
    }

    fn open(self: Box<Self>) -> VortexResult<ScanExec> {
        Ok(plan_stream::open(self.state, self.planned))
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(Box::new(PlanPartition(self)) as PartitionRef)]).boxed()
    }
}

/// A scan of a [`PlanDataSource`] as its one partition, which reads one stream of the opened scan.
struct PlanPartition(Box<PlanScan>);

impl Partition for PlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        Ok(Box::pin(self.0.open()?.stream()))
    }
}
