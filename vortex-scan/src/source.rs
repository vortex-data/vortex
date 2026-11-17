// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source inputs and byte access for scans.

use std::any::Any;
use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::execution::SegmentLocation;
use crate::execution::cache::ScanCache;
use crate::plan::LazyPlanRef;
use crate::request::ScanRequest;

/// Identifier for a single physical segment referenced by a layout.
///
/// Segment ids are local to a file or segment source. The file footer maps ids to physical offsets;
/// custom storage systems may map them to object-store keys or other random-access locations.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl TryFrom<usize> for SegmentId {
    type Error = VortexError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::from(u32::try_from(value)?))
    }
}

impl Deref for SegmentId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides bytes to scan sources.
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;
}

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
    /// Fetches bytes by source-local segment identifier.
    pub segments: Arc<dyn SegmentSource>,
    /// State shared by repeated scans of this input.
    pub cache: Option<ScanCache>,
}

/// A sendable stream of partitions.
pub type PartitionStream = BoxStream<'static, VortexResult<PartitionRef>>;

/// Opens a Vortex [`DataSource`] from a URI.
///
/// Configuration can be passed via the URI query parameters, similar to JDBC / ADBC.
/// Providers can be registered with the [`VortexSession`] to support additional URI schemes.
#[async_trait]
pub trait DataSourceOpener: 'static {
    /// Attempt to open a new data source from a URI.
    async fn open(&self, uri: String, session: &VortexSession) -> VortexResult<DataSourceRef>;
}

/// Supports deserialization of a Vortex [`DataSource`] on a remote worker.
#[async_trait]
pub trait DataSourceRemote: 'static {
    /// Attempt to deserialize the source.
    fn deserialize_data_source(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<DataSourceRef>;
}

/// A reference-counted data source.
pub type DataSourceRef = Arc<dyn DataSource>;

/// A data source represents a streamable dataset that can be scanned with projection and filter
/// expressions. Each scan produces partitions that can be executed in parallel to read data. Each
/// partition can be serialized for remote execution.
///
/// The DataSource may be used multiple times to create multiple scans, whereas each scan and each
/// partition of a scan can only be consumed once.
///
/// Partitions have indices. Partition index is stable throughout DataSource's
/// lifetime. For every scan requested on the DataSource, the index will stay
/// the same.
/// However, this means you should create another instance of a DataSource if
/// your environment changes e.g. you have a glob and another file is added to
/// the filesystem this glob references.
/// See MultiFileDataSource in vortex-file/src/multi/mod.rs
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the un-filtered source.
    fn row_count(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Returns an estimate of the byte size of the un-filtered source.
    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Serialize the [`DataSource`] to pass to a remote worker.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Deserialize a partition that was previously serialized from a compatible data source.
    fn deserialize_partition(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        let _ = (data, session);
        vortex_bail!("DataSource does not support deserialization")
    }

    /// Returns a scan over the source.
    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef>;

    /// Returns the statistics for a given field.
    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet>;

    /// Returns a stored aggregate partial over the unfiltered source field.
    ///
    /// The scalar has the aggregate's `partial_dtype`; callers merge partials and finalize
    /// through the requested aggregate. Identity includes bound options. Missing partials,
    /// incomplete coverage, and incompatible aggregates return [`Precision::Absent`], as
    /// does the default implementation. Finalized legacy statistics are not partials.
    ///
    /// Lookup must not scan payloads. Precision describes the partial state; consumers must
    /// respect the aggregate's approximation contract. Later scan filters and limits do not
    /// change this source-level partial. A stored null partial is distinct from absence.
    async fn field_statistic(
        &self,
        _field_path: &FieldPath,
        _aggregate_fn: &AggregateFnRef,
    ) -> VortexResult<Precision<Scalar>> {
        Ok(Precision::Absent)
    }
}

/// A boxed data source scan.
pub type DataSourceScanRef = Box<dyn DataSourceScan>;

/// A data source scan produces partitions that can be executed to read data from the source.
pub trait DataSourceScan: 'static + Send {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the total number of partitions the scan will produce.
    fn partition_count(&self) -> Precision<usize>;

    /// Returns a stream of partitions to be processed.
    fn partitions(self: Box<Self>) -> PartitionStream;
}

/// A reference-counted partition.
pub type PartitionRef = Box<dyn Partition>;

/// A partition represents a unit of work that can be executed to produce a stream of arrays.
pub trait Partition: 'static + Send {
    /// Downcast the partition to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Some unique identifier for partition.
    /// If you have an instance of a DataSource, the indices of emitted
    /// partitions will stay stable for every scan in this DataSource.
    fn index(&self) -> usize;

    /// Returns an estimate of the row count for this partition.
    fn row_count(&self) -> Precision<u64>;

    /// Returns an estimate of the byte size for this partition.
    fn byte_size(&self) -> Precision<u64>;

    /// Serialize this partition for a remote worker.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Executes the partition, returning an array stream.
    ///
    /// This method must be fast. The returned stream should be lazy — all non-trivial work
    /// (I/O, decoding, filtering) must be deferred to when the stream is polled. Expensive
    /// operations should be spawned onto the runtime to enable parallel execution across
    /// threads.
    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream>;
}
