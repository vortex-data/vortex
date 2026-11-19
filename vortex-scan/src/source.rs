// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source inputs and byte access for scans, and the [`DataSource`] that scans them.

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future;
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
use vortex_io::request::SegmentReader;
pub use vortex_io::segments::SegmentFuture;
pub use vortex_io::segments::SegmentId;
pub use vortex_io::segments::SegmentSource;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionPlan;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::execution::SegmentLocation;
use crate::execution::cache::ScanCache;
use crate::execution::driver::ScanDriver;
use crate::plan::LazyPlanRef;
use crate::prepare::ScanBuilder;
use crate::selection::Selection;

/// The plan and byte source used to prepare a scan.
///
/// The lazy root defers lowering until preparation runs on a compute task. Segment identifiers
/// are local to this input. Reuse a cache only while the plan, dtype, locations, and source
/// describe the same data.
#[derive(Clone)]
pub struct ScanInput {
    /// The dtype of the root plan.
    pub dtype: DType,
    /// The root plan, initialized without reading payloads.
    pub root: LazyPlanRef,
    /// Byte locations indexed by source-local segment identifiers.
    pub locations: Arc<[SegmentLocation]>,
    /// How the session's I/O scheduler reads segments by source-local identifier.
    pub reader: SegmentReader,
    /// State shared by repeated scans of this input.
    pub cache: Option<ScanCache>,
}

/// A [`DataSource`] over one [`ScanInput`], scanned by the planner-driven executor.
///
/// This is the standard way to expose a source that can name its segments, whatever its format:
/// a Vortex file, a Parquet file whose column chunks are placed segments, or any store that can
/// lower its data to plans. Every scan of the source shares the input's [`ScanCache`], so scans
/// that overlap in time plan the input and register its reader once.
///
/// A scan has one partition covering the request's row range. Its
/// [`plan`](Partition::plan) is the scan builder, for engines that drive the executor
/// themselves; its [`execute`](Partition::execute) runs that builder with a [`ScanDriver`] on
/// the session's runtime and applies the request's limit.
#[derive(Clone)]
pub struct PlannedDataSource {
    session: VortexSession,
    input: ScanInput,
    row_count: Precision<u64>,
    byte_size: Precision<u64>,
}

impl PlannedDataSource {
    /// A source scanning `input` with `session`. Without a cache, the input gets a fresh one that
    /// all of the source's scans share.
    pub fn new(session: VortexSession, mut input: ScanInput) -> Self {
        input.cache.get_or_insert_default();
        Self {
            session,
            input,
            row_count: Precision::Absent,
            byte_size: Precision::Absent,
        }
    }

    /// Sets the exact row count of the input, which scans and partitions report.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Precision::exact(row_count);
        self
    }

    /// Sets an estimate of the input's byte size.
    pub fn with_byte_size(mut self, byte_size: Precision<u64>) -> Self {
        self.byte_size = byte_size;
        self
    }

    /// The input this source scans.
    pub fn input(&self) -> &ScanInput {
        &self.input
    }

    /// The session scans of this source run with.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

#[async_trait]
impl DataSource for PlannedDataSource {
    fn dtype(&self) -> &DType {
        &self.input.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn byte_size(&self) -> Precision<u64> {
        self.byte_size
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let projection = request
            .projection
            .bind(&self.input.dtype)?
            .optimize_recursive()?;
        let filter = request
            .filter
            .map(|filter| filter.bind(&self.input.dtype)?.optimize_recursive())
            .transpose()?;
        let dtype = projection.dtype().clone();
        let row_range = match (request.row_range, self.row_count.as_exact()) {
            (Some(rows), Some(count)) => Some(rows.start.min(count)..rows.end.min(count)),
            (rows, _) => rows,
        };
        // The scan's one partition has index 0.
        let selected = request
            .partition_range
            .as_ref()
            .is_none_or(|range| range.contains(&0))
            && request.partition_selection.row_count_in_range(&(0..1)) == 1;
        let empty = request.limit == Some(0) || row_range.as_ref().is_some_and(Range::is_empty);
        let partition = (selected && !empty).then(|| PlannedPartition {
            session: self.session.clone(),
            input: self.input.clone(),
            row_count: self.row_count,
            projection,
            filter,
            row_range,
            selection: request.selection,
            ordered: request.ordered,
            limit: request.limit,
            split_concurrency: request.split_concurrency,
        });
        Ok(Box::new(PlannedScan { dtype, partition }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

struct PlannedScan {
    dtype: DType,
    partition: Option<PlannedPartition>,
}

impl DataSourceScan for PlannedScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(usize::from(self.partition.is_some()))
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter(
            self.partition
                .map(|partition| Ok(Box::new(partition) as PartitionRef)),
        )
        .boxed()
    }
}

/// The rows of a [`PlannedDataSource`] that one scan reads.
struct PlannedPartition {
    session: VortexSession,
    input: ScanInput,
    row_count: Precision<u64>,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    selection: Selection,
    ordered: bool,
    limit: Option<u64>,
    split_concurrency: Option<NonZeroUsize>,
}

impl PlannedPartition {
    fn builder(self) -> ScanBuilder {
        let mut builder = ScanBuilder::new(self.session, self.input)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_selection(self.selection);
        if let Some(rows) = self.row_range {
            builder = builder.with_row_range(rows);
        }
        builder
    }
}

impl Partition for PlannedPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = match (&self.row_range, self.row_count.as_exact()) {
            (Some(rows), _) => rows.clone(),
            (None, Some(count)) => 0..count,
            (None, None) => return Precision::Absent,
        };
        let rows = self.selection.row_count_in_range(&rows);
        let rows = self.limit.map_or(rows, |limit| rows.min(limit));
        if self.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn plan(self: Box<Self>) -> VortexResult<PartitionPlan> {
        Ok(PartitionPlan::Scan(Box::new(self.builder())))
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let dtype = self.projection.dtype().clone();
        let (ordered, limit, max_in_flight) = (self.ordered, self.limit, self.split_concurrency);
        let driver =
            ScanDriver::new(self.session.handle(), 1, ordered).with_max_in_flight(max_in_flight);
        let arrays = driver.run(self.builder());
        let arrays = match limit {
            Some(limit) => limit_arrays(arrays, limit),
            None => arrays,
        };
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, arrays,
        )))
    }
}

/// Ends `arrays` after its first `limit` rows.
fn limit_arrays(
    arrays: BoxStream<'static, VortexResult<ArrayRef>>,
    limit: u64,
) -> BoxStream<'static, VortexResult<ArrayRef>> {
    arrays
        .scan(limit, |remaining, array| {
            if *remaining == 0 {
                return future::ready(None);
            }
            let array = array.and_then(|array| {
                let len = array.len() as u64;
                if len <= *remaining {
                    *remaining -= len;
                    Ok(array)
                } else {
                    let keep = usize::try_from(*remaining).unwrap_or(usize::MAX);
                    *remaining = 0;
                    array.slice(0..keep)
                }
            });
            future::ready(Some(array))
        })
        .boxed()
}

#[cfg(test)]
mod tests;
