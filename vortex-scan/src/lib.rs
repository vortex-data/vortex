// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! The Vortex Scan API implements an abstract table scan interface that can be used to
//! read data from various data sources.
//!
//! It supports arbitrary projection expressions, filter expressions, and limit pushdown as well
//! as mechanisms for parallel and distributed execution via partitions.
//!
//! A [`DataSourceScan`] can also be run without partitions. The host decides parallelism: it may
//! [`split`](DataSourceScan::split) the scan into independent pieces (for example to ship them to
//! other machines, or to keep row order across threads) and [`open`](DataSourceScan::open) each
//! piece into a [`ScanPool`]. Any number of workers, created with [`ScanPool::worker`], then share
//! the pool's work. A worker is a stream polled inline by its caller.
//!
//! The API is currently under development and may change in future releases, however we hope to
//! stabilize into stable C ABI for use within foreign language bindings.
//!
//! If you are looking to scan Vortex files or layouts, the Vortex implementation of the Scan API
//! is lowered by `vortex-layout` into this crate's plans and executor.
//!
//! ## Open Issues
//!
//! * We probably want to make the DataSource serializable as well, so that we can share
//!   source-level state with workers, separately from partition serialization.
//! * We should add a way for the client to negotiate capabilities with the data source, for
//!   example which encodings it knows about.

pub mod arrays;
pub mod execution;
mod partition_pool;
pub mod plan;
pub mod planning;
pub mod prepare;
pub mod row_mask;
pub mod selection;
pub mod source;
pub mod strict_sorted_buffer;

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use selection::Selection;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

pub use crate::partition_pool::PartitionPool;

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

/// A request to scan a data source.
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Projection expression. Defaults to `root()` which returns all columns.
    pub projection: Expression,
    /// Filter expression, `None` implies no filter.
    pub filter: Option<Expression>,
    /// The per-partition row range to read. Row range will be applied
    /// over every partition you scan.
    pub row_range: Option<Range<u64>>,
    /// The per-partition row selection to read. Row selection will be applied
    /// over every partition you scan.
    pub selection: Selection,
    /// Partition selection to scan, which allows readers to skip unwanted partitions.
    pub partition_selection: Selection,
    /// Partition range to scan, which allows readers to skip unwanted partitions.
    pub partition_range: Option<Range<u64>>,
    /// Whether the scan should preserve row order. If false, the scan may produce rows in any
    /// order, for example to enable parallel execution across partitions.
    pub ordered: bool,
    /// Optional limit on the number of rows returned, applied after filtering and row selection.
    ///
    /// For an unordered scan the limit is global: partitions share one budget, so together they
    /// return no more than `limit` rows. An ordered scan cannot share a budget whose reservation
    /// order is completion order, so each partition applies the limit locally and the caller must
    /// trim the concatenated result.
    pub limit: Option<u64>,
    /// The most splits of each partition to decode at once, in all.
    ///
    /// `None` keeps the reader's default, which scales with the machine's cores. Each split being
    /// decoded holds the decoded chunks of every column it overlaps, so a scan that runs alongside
    /// others, such as one task of several on a machine, can set this to bound its memory. It does
    /// not bound I/O: segment reads for later splits may still be coalesced ahead of them.
    pub split_concurrency: Option<NonZeroUsize>,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            projection: root(),
            filter: None,
            row_range: None,
            selection: Selection::default(),
            partition_selection: Selection::default(),
            ordered: false,
            limit: None,
            partition_range: None,
            split_concurrency: None,
        }
    }
}

/// A boxed data source scan.
pub type DataSourceScanRef = Box<dyn DataSourceScan>;

/// A data source scan produces partitions that can be executed to read data from the source.
///
/// A scan is consumed once: by [`partitions`](Self::partitions), or by [`split`](Self::split) and
/// [`open`](Self::open).
pub trait DataSourceScan: 'static + Send + IntoDataSourceScanRef {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the total number of partitions the scan will produce.
    fn partition_count(&self) -> Precision<usize>;

    /// Returns a stream of partitions to be processed.
    fn partitions(self: Box<Self>) -> PartitionStream;

    /// Whether a single worker of this scan's [`ScanPool`] yields rows in row order.
    ///
    /// Defaults to `true`, so that a source that does not say keeps its order.
    fn ordered(&self) -> bool {
        true
    }

    /// Estimates the rows this scan returns, from plan-time state only.
    fn row_count(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Estimates the bytes this scan reads, before any filter, from plan-time state only.
    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Cuts the scan into at most `n` independent scans, in row order, from metadata only.
    ///
    /// Together the pieces read exactly what this scan reads. A source may return fewer pieces,
    /// and returns the scan itself when it cannot be cut (the default). Never returns an empty
    /// list.
    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        let _ = n;
        Ok(vec![self.into_data_source_scan_ref()])
    }

    /// Opens the scan for execution by any number of workers.
    ///
    /// Must be cheap: all I/O happens when the pool's workers are polled. The default is a
    /// [`PartitionPool`] over [`partitions`](Self::partitions).
    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        Ok(Arc::new(PartitionPool::new(
            self.into_data_source_scan_ref(),
        )))
    }
}

/// Converts a boxed scan into a [`DataSourceScanRef`].
///
/// Implemented for every sized [`DataSourceScan`], so that its default methods can return the
/// scan itself.
pub trait IntoDataSourceScanRef {
    /// Returns this scan as a [`DataSourceScanRef`].
    fn into_data_source_scan_ref(self: Box<Self>) -> DataSourceScanRef;
}

impl<T: DataSourceScan> IntoDataSourceScanRef for T {
    fn into_data_source_scan_ref(self: Box<Self>) -> DataSourceScanRef {
        self
    }
}

/// A reference-counted, opened scan.
pub type ScanPoolRef = Arc<dyn ScanPool>;

/// An opened scan whose work any number of workers share.
///
/// Each unit of work is read by exactly one worker.
pub trait ScanPool: 'static + Send + Sync {
    /// The dtype of the arrays the workers return.
    fn dtype(&self) -> &DType;

    /// Returns a new worker: a stream of the scan's arrays that the calling thread polls inline.
    ///
    /// Workers may be created at any time and polled from any thread; a worker ends once the
    /// pool has no work left for it. For an ordered scan, a single worker yields rows in row
    /// order; several workers give no global order.
    fn worker(self: Arc<Self>) -> SendableArrayStream;

    /// True once every unit has been handed to a worker (workers may still be producing).
    ///
    /// A host uses it to stop adding threads to a scan that has nothing left to hand out.
    fn is_exhausted(&self) -> bool;

    /// Fraction in `[0, 1]` of the scan's units, weighted by row count where known, handed out
    /// so far.
    fn progress(&self) -> f64;
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

/// Expressions referring to the global scan row index.
pub mod row_idx {
    pub use crate::plan::builtins::row_idx::expr::*;
}

#[cfg(test)]
mod test;
