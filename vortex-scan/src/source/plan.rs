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
use vortex_array::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::ScanInput;
use super::bind::BoundRequest;
use super::plan_stream;
use super::plan_stream::Planned;
use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::limit::RowLimit;
use crate::prepare::ScanBuilder;
use crate::prepare::SharedFilterPrep;
use crate::selection::Selection;
use crate::stream::ScanExec;

/// Whether a child's own statistics, such as a file footer's, prove a filter, bound to the
/// child's dtype, false for every row of the child.
pub type FilePruner = Arc<dyn Fn(&BoundExpression) -> VortexResult<bool> + Send + Sync>;

/// The child of a [`PlanDataSource`]: its [`ScanInput`] and what the source knows of it without
/// reading payloads.
#[derive(Clone)]
pub struct PlanChild {
    input: ScanInput,
    row_count: Option<u64>,
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
            pruner: None,
        }
    }

    /// Sets the child's exact row count. Without it, the source lowers the child's root plan to
    /// learn the count when a scan needs it.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
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
/// Scans are unordered: a request for an ordered scan is an error. Making a scan prepares the
/// child and cuts its rows into parts on the scan's chunk grid, reading nothing; a scan whose
/// filter the child's own statistics rule out has no parts. [`split`](DataSourceScan::split) deals
/// the parts out to pieces of whole grid blocks, for a host that gives each thread a piece.
/// [`open`](DataSourceScan::open) runs a scan on V2's planners and
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
        let bound = match &request.bind_cache {
            Some(cache) => cache.bind(dtype, &request.projection, request.filter.as_ref())?,
            None => Arc::new(BoundRequest::new(
                dtype,
                &request.projection,
                request.filter.as_ref(),
            )?),
        };
        let projection = bound.projection.clone();
        let filter = bound.filter.clone();
        let selected = request
            .partition_range
            .as_ref()
            .is_none_or(|range| range.contains(&0))
            && request.partition_selection.row_count_in_range(&(0..1)) == 1;
        let scans_rows =
            request.limit != Some(0) && !request.row_range.as_ref().is_some_and(Range::is_empty);
        let rows = if selected && scans_rows && !self.child.is_pruned(filter.as_ref())? {
            Some(clamp_rows(request.row_range, self.child.rows()?))
        } else {
            None
        };
        let state = ScanState {
            session: self.session.clone(),
            child: self.child.clone(),
            dtype: projection.dtype().clone(),
            projection,
            filter,
            filter_prep: request
                .bind_cache
                .is_some()
                .then(|| Arc::clone(&bound.filter_prep)),
            selection: request.selection,
            limit: request.limit.map(RowLimit::new),
        };
        let planned = Planned::new(&state, rows)?;
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
    /// The filter's rewrites, shared with the scans of the other sources of a [`BindCache`].
    ///
    /// [`BindCache`]: super::BindCache
    filter_prep: Option<Arc<SharedFilterPrep>>,
    selection: Selection,
    /// The scan's row budget, which every piece of it shares.
    pub(super) limit: Option<RowLimit>,
}

impl ScanState {
    /// A builder of this scan's `rows` of the child.
    pub(super) fn builder(&self, rows: Range<u64>) -> ScanBuilder {
        let builder = ScanBuilder::new(self.session.clone(), self.child.input.clone())
            .with_projection(self.projection.clone())
            .with_some_filter(self.filter.clone())
            .with_selection(self.selection.clone())
            .with_row_range(rows);
        match &self.filter_prep {
            Some(prep) => builder.with_filter_prep(Arc::clone(prep)),
            None => builder,
        }
    }
}

/// A scan of a [`PlanDataSource`], or a piece of one.
struct PlanScan {
    state: Arc<ScanState>,
    /// The scan's parts, cut when the scan was made.
    planned: Planned,
}

impl DataSourceScan for PlanScan {
    fn dtype(&self) -> &DType {
        &self.state.dtype
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        // A limit holds across the whole scan, so a limited scan is not split.
        if n.get() == 1 || self.planned.len() == 0 || self.state.limit.is_some() {
            return Ok(vec![self]);
        }
        let state = self.state;
        Ok(self
            .planned
            .split(n.get())
            .into_iter()
            .map(|planned| {
                Box::new(PlanScan {
                    state: Arc::clone(&state),
                    planned,
                }) as DataSourceScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanExec> {
        Ok(plan_stream::open(&self.state, self.planned))
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
