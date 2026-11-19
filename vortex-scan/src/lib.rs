// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! The Vortex Scan API implements an abstract table scan interface that can be used to
//! read data from various data sources.
//!
//! It supports arbitrary projection expressions, filter expressions, and limit pushdown as well
//! as mechanisms for parallel and distributed execution via partitions.
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
/// Every partition has an [`index`](Partition::index) naming the unit of the source it reads,
/// which is what [`ScanRequest::partition_range`] and [`ScanRequest::partition_selection`]
/// select. A [`PlanDataSource`](source::PlanDataSource) is one unit with index 0, and a source
/// with several units, such as one over several files, gives each its own index. The index is
/// stable for the lifetime of the DataSource, so every scan of it gives a unit the same index. It
/// is not unique within a scan: a unit may be split into several partitions that share its index.
///
/// Because indices are stable, create another DataSource when the data it covers changes, for
/// example when a file is added to the directory that a glob names. See `MultiFileDataSource` in
/// `vortex-file`.
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
    /// The range of rows to read from each selected unit, in that unit's own row numbering.
    ///
    /// Units are what [`Partition::index`] names: for a source over several files, each file is a
    /// unit and the range applies to each file's rows. A unit that a scan splits into several
    /// partitions applies the range once across them, so each row is read by one partition.
    pub row_range: Option<Range<u64>>,
    /// The rows to read from each selected unit, by row index in that unit's own row numbering.
    ///
    /// The indices are not relative to [`row_range`](Self::row_range): a scan reads the rows that
    /// are both in the row range and selected. Like the row range, the selection applies to every
    /// selected unit, and once across the partitions of a unit that a scan splits.
    pub selection: Selection,
    /// The units to scan, by [`Partition::index`], which allows readers to skip unwanted units.
    pub partition_selection: Selection,
    /// The range of units to scan, by [`Partition::index`], which allows readers to skip unwanted
    /// units.
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
    /// The number of partitions the caller would like the scan to produce, as a hint.
    ///
    /// A source may return more or fewer partitions than this, for example because it has more
    /// units than the hint or because a unit is too small to be worth splitting. `None` lets the
    /// source choose: a [`PlanDataSource`](source::PlanDataSource) then uses the machine's
    /// available parallelism. The hint only affects scans that a source is free to split, which
    /// are unordered and have no limit. Defaults to `None`.
    pub target_partitions: Option<NonZeroUsize>,
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
            target_partitions: None,
        }
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

    /// The index of the unit of the source this partition reads, which
    /// [`ScanRequest::partition_range`] and [`ScanRequest::partition_selection`] select.
    ///
    /// For a [`PlanDataSource`](source::PlanDataSource) it is 0, since the source is one unit; a
    /// source with several units gives each its own index. The index is stable for the lifetime of
    /// the [`DataSource`], but not unique within a scan: a unit may be split into several
    /// partitions that share its index.
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
