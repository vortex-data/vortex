// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PlanDataSource`]: a [`DataSource`] over plan children, scanned by the planner-driven
//! executor.
//!
//! Nothing here is about one format. A child is a [`ScanInput`], a plan over placed segments,
//! from a Vortex file, a Parquet file whose column chunks are segments, or any store that lowers
//! its data to plans; a deferred child is anything that expands into children, such as a file to
//! open, a table format's manifest, or a partition to list.

use std::any::Any;
use std::collections::VecDeque;
use std::future;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::ScanInput;
use super::bind_cache::BindCache;
use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::execution::driver::ScanDriver;
use crate::pool::PartitionPool;
use crate::pool::RowLimit;
use crate::pool::ScanPoolRef;
use crate::pool::trim_to_limit;
use crate::prepare::ScanBuilder;
use crate::selection::Selection;

/// Whether a child's own statistics, such as a file footer's, prove a filter, bound to the
/// child's dtype, false for every row of the child.
pub type FilePruner = Arc<dyn Fn(&BoundExpression) -> VortexResult<bool> + Send + Sync>;

/// A child of a [`PlanDataSource`] whose plan is ready to prepare: its [`ScanInput`] and what the
/// source knows of it without reading payloads.
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
    pub(super) fn rows(&self) -> VortexResult<u64> {
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

/// A child of a [`PlanDataSource`] that a scan expands when it reaches it: a file to open, a
/// manifest entry to read, a table partition to list.
///
/// Expanding is work like any other unit of an opened scan: the worker that takes the child
/// expands it and queues the work of what it expanded into, for any worker to take.
#[async_trait]
pub trait DeferredPlanChild: 'static + Send + Sync {
    /// The children this child stands for, in row order. They may be deferred in turn, as a
    /// manifest's children are. An empty list skips the child.
    async fn expand(&self) -> VortexResult<Vec<PlanSourceChild>>;
}

/// A child of a [`PlanDataSource`].
#[derive(Clone)]
pub enum PlanSourceChild {
    /// A child whose plan is ready to prepare.
    Ready(PlanChild),
    /// A child that a scan expands when it reaches it, so a source over many children reads only
    /// the metadata its scans need.
    Deferred {
        /// Expands the child.
        child: Arc<dyn DeferredPlanChild>,
        /// The child's size in bytes, if known.
        byte_size: Option<u64>,
    },
}

/// A [`DataSource`] over plan children, scanned by the planner-driven executor.
///
/// This is the standard way to expose sources that can lower their data to plans over placed
/// segments, whatever their format. Every scan of a ready child shares the child's
/// [`ScanCache`], so scans that overlap in time plan it once.
///
/// A scan's request applies to each child: its row range and row selection are in the child's
/// own rows. A scan reads its children in order, and a deferred child's children in its place.
///
/// # Splitting and opening
///
/// The work of a scan is plan work: chunks of a prepared plan, and the expansion of deferred
/// children. [`split`](DataSourceScan::split) does not cut a scan yet: it returns the scan whole.
/// [`open`](DataSourceScan::open) reads the scan's partitions through a [`PartitionPool`], each
/// worker taking one partition at a time.
///
/// # Partitions
///
/// A scan has one partition per child, whose [`index`](Partition::index) is the child's ordinal,
/// so a request's `partition_range` and `partition_selection` select children. A partition runs
/// its child, or each child a deferred one expands into, with a [`ScanDriver`] on the session's
/// runtime.
///
/// [`ScanCache`]: crate::prepare::ScanCache
#[derive(Clone)]
pub struct PlanDataSource {
    session: VortexSession,
    dtype: DType,
    children: Arc<[PlanSourceChild]>,
    row_count: Precision<u64>,
    byte_size: Precision<u64>,
}

impl PlanDataSource {
    /// A source scanning one ready `input` with `session`.
    pub fn new(session: VortexSession, input: ScanInput) -> Self {
        Self::from_child(session, PlanChild::new(input))
    }

    /// A source scanning one ready `child` with `session`.
    pub fn from_child(session: VortexSession, child: PlanChild) -> Self {
        let dtype = child.input.dtype.clone();
        let row_count = child.row_count.map_or(Precision::Absent, Precision::exact);
        Self {
            session,
            dtype,
            children: Arc::new([PlanSourceChild::Ready(child)]),
            row_count,
            byte_size: Precision::Absent,
        }
    }

    /// A source scanning `children` in order with `session`. Every child must have `dtype`; a
    /// scan that expands a child of another dtype fails.
    pub fn from_children(
        session: VortexSession,
        dtype: DType,
        children: Vec<PlanSourceChild>,
    ) -> Self {
        let mut rows = Some(0u64);
        let mut bytes = Some(0u64);
        let mut known_rows = 0;
        let mut known_files = 0;
        for file in &children {
            match file {
                PlanSourceChild::Ready(file) => {
                    match file.row_count {
                        Some(count) => {
                            known_rows += count;
                            known_files += 1;
                        }
                        None => rows = None,
                    }
                    bytes = bytes.map(|bytes| bytes + file.byte_size());
                }
                PlanSourceChild::Deferred { byte_size, .. } => {
                    rows = None;
                    bytes = bytes.zip(*byte_size).map(|(bytes, size)| bytes + size);
                }
            }
        }
        // Deferred children are estimated at the average row count of the ready ones.
        let row_count = match rows {
            Some(rows) => Precision::exact(rows + known_rows),
            None if known_files > 0 => {
                Precision::inexact(known_rows / known_files * children.len() as u64)
            }
            None => Precision::Absent,
        };
        Self {
            session,
            dtype,
            children: children.into(),
            row_count,
            byte_size: bytes.map_or(Precision::Absent, Precision::inexact),
        }
    }

    /// Sets the exact row count of a one-child source, which scans and partitions report.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        if let [PlanSourceChild::Ready(file)] = &*self.children {
            self.children = Arc::new([PlanSourceChild::Ready(
                file.clone().with_row_count(row_count),
            )]);
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
    pub fn children(&self) -> &[PlanSourceChild] {
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
        for (ordinal, file) in self.children.iter().enumerate().filter(|_| scans_rows) {
            let index = ordinal as u64;
            let selected = request
                .partition_range
                .as_ref()
                .is_none_or(|range| range.contains(&index))
                && request
                    .partition_selection
                    .row_count_in_range(&(index..index + 1))
                    == 1;
            if !selected {
                continue;
            }
            let rows = match file {
                PlanSourceChild::Ready(file) => {
                    if file.is_pruned(filter.as_ref())? {
                        continue;
                    }
                    file.row_count
                        .map(|count| clamp_rows(request.row_range.clone(), count))
                }
                PlanSourceChild::Deferred { .. } => None,
            };
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
            source_dtype: self.dtype.clone(),
            children: Arc::clone(&self.children),
            dtype: projection.dtype().clone(),
            projection,
            filter,
            row_range: request.row_range,
            selection: request.selection,
            ordered: request.ordered,
            limit: request.limit,
            split_concurrency: request.split_concurrency,
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

/// What every piece, partition and pool of one scan of a [`PlanDataSource`] shares.
pub(super) struct ScanState {
    pub(super) session: VortexSession,
    source_dtype: DType,
    children: Arc<[PlanSourceChild]>,
    /// The dtype of the scan's output.
    pub(super) dtype: DType,
    pub(super) projection: BoundExpression,
    pub(super) filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    pub(super) selection: Selection,
    pub(super) ordered: bool,
    pub(super) limit: Option<u64>,
    pub(super) split_concurrency: Option<NonZeroUsize>,
}

impl ScanState {
    /// The source's child at `ordinal`.
    pub(super) fn child(&self, ordinal: usize) -> &PlanSourceChild {
        &self.children[ordinal]
    }

    /// The child at `ordinal` if it is ready.
    pub(super) fn ready(&self, ordinal: usize) -> Option<&PlanChild> {
        match &self.children[ordinal] {
            PlanSourceChild::Ready(child) => Some(child),
            PlanSourceChild::Deferred { .. } => None,
        }
    }

    /// Admits a child that expanding produced: `None` when its statistics rule out the filter,
    /// and an error when its dtype is not the source's.
    pub(super) fn admit(&self, child: PlanChild) -> VortexResult<Option<PlanChild>> {
        if child.input.dtype != self.source_dtype {
            vortex_bail!(
                "a child of the source has dtype {}, but the source has dtype {}",
                child.input.dtype,
                self.source_dtype
            );
        }
        if child.is_pruned(self.filter.as_ref())? {
            return Ok(None);
        }
        Ok(Some(child))
    }

    /// Expands `children` depth-first, in order, into the admitted ready children they stand for,
    /// expanding each deferred child only once the children before it are taken.
    ///
    /// Each child comes with its share of the scan's work, which a deferred child divides evenly
    /// among the children it expands into. The share of a child that is pruned, or that expands
    /// into nothing, goes to `skipped`.
    pub(super) fn expand_in_order(
        self: &Arc<Self>,
        children: Vec<(PlanSourceChild, f64)>,
        skipped: Arc<dyn Fn(f64) + Send + Sync>,
    ) -> BoxStream<'static, VortexResult<(PlanChild, f64)>> {
        let state = Arc::clone(self);
        stream::try_unfold(VecDeque::from(children), move |mut queue| {
            let state = Arc::clone(&state);
            let skipped = Arc::clone(&skipped);
            async move {
                while let Some((child, share)) = queue.pop_front() {
                    match child {
                        PlanSourceChild::Ready(child) => match state.admit(child)? {
                            Some(child) => return Ok(Some(((child, share), queue))),
                            None => skipped(share),
                        },
                        PlanSourceChild::Deferred { child, .. } => {
                            let children = child.expand().await?;
                            if children.is_empty() {
                                skipped(share);
                            }
                            let share = share / children.len().max(1) as f64;
                            for child in children.into_iter().rev() {
                                queue.push_front((child, share));
                            }
                        }
                    }
                }
                Ok(None)
            }
        })
        .boxed()
    }

    /// Runs `rows` of `child`, or the request's rows of it when `None`, with a [`ScanDriver`].
    fn drive(
        &self,
        child: &PlanChild,
        rows: Option<Range<u64>>,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let rows = match rows.map_or_else(|| self.child_rows(child), Ok) {
            Ok(rows) => rows,
            Err(err) => return stream::once(future::ready(Err(err))).boxed(),
        };
        if rows.is_empty() {
            return stream::empty().boxed();
        }
        ScanDriver::new(self.session.handle(), 1, self.ordered)
            .with_max_in_flight(self.split_concurrency)
            .run(self.builder(child, rows))
    }

    /// The rows of `child` this scan reads: the request's row range, clamped to the child.
    pub(super) fn child_rows(&self, child: &PlanChild) -> VortexResult<Range<u64>> {
        Ok(clamp_rows(self.row_range.clone(), child.rows()?))
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
    /// until it is open.
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
    fn unit_bytes(&self, unit: &Unit) -> Option<u64> {
        match &self.state.children[unit.child] {
            PlanSourceChild::Ready(file) => {
                let bytes = file.byte_size();
                match (&unit.rows, file.row_count) {
                    (Some(rows), Some(count)) if count > 0 => Some(
                        u64::try_from(
                            u128::from(bytes) * u128::from(rows.end - rows.start)
                                / u128::from(count),
                        )
                        .unwrap_or(u64::MAX),
                    ),
                    _ => Some(bytes),
                }
            }
            PlanSourceChild::Deferred { byte_size, .. } => *byte_size,
        }
    }
}

impl DataSourceScan for PlanScan {
    fn dtype(&self) -> &DType {
        &self.state.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(self.units.len())
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        // An unordered scan's partitions share one budget; an ordered scan's each apply the limit
        // locally, and the caller trims the concatenated result.
        let shared = (!self.state.ordered)
            .then(|| self.state.limit.map(RowLimit::new))
            .flatten();
        let state = self.state;
        stream::iter(self.units.into_iter().map(move |unit| {
            let limit = shared.clone().or_else(|| state.limit.map(RowLimit::new));
            Ok(Box::new(PlanPartition {
                state: Arc::clone(&state),
                unit,
                limit,
            }) as PartitionRef)
        }))
        .boxed()
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
        self.units
            .iter()
            .map(|unit| self.unit_bytes(unit))
            .sum::<Option<u64>>()
            .map_or(Precision::Absent, Precision::inexact)
    }

    fn split(self: Box<Self>, _n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        Ok(vec![self])
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        Ok(Arc::new(PartitionPool::new(self)))
    }
}

/// One child's rows of a scan of a [`PlanDataSource`].
struct PlanPartition {
    state: Arc<ScanState>,
    unit: Unit,
    limit: Option<RowLimit>,
}

impl Partition for PlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.unit.child
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = match (&self.unit.rows, self.state.ready(self.unit.child)) {
            (Some(rows), _) => rows.clone(),
            (None, Some(file)) if file.row_count.is_some() => match self.state.child_rows(file) {
                Ok(rows) => rows,
                Err(_) => return Precision::Absent,
            },
            _ => return Precision::Absent,
        };
        let rows = self.state.selection.row_count_in_range(&rows);
        let rows = self.state.limit.map_or(rows, |limit| rows.min(limit));
        if self.state.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let PlanPartition { state, unit, limit } = *self;
        let dtype = state.dtype.clone();
        // A ready child runs as it is; a deferred one runs each child it expands into, in order.
        let children = match state.ready(unit.child) {
            Some(child) => stream::iter([Ok((child.clone(), unit.rows))]).boxed(),
            None => state
                .expand_in_order(
                    vec![(state.child(unit.child).clone(), 0.0)],
                    Arc::new(|_| {}),
                )
                .map_ok(|(child, _)| (child, None))
                .boxed(),
        };
        let driver = Arc::clone(&state);
        let arrays = children
            .map_ok(move |(child, rows)| driver.drive(&child, rows))
            .try_flatten()
            .boxed();
        let arrays = match limit {
            Some(limit) => trim_to_limit(arrays, limit),
            None => arrays,
        };
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, arrays,
        )))
    }
}
