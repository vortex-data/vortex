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
//! can be found in the `vortex-layout` crate.
//!
//! ## Open Issues
//!
//! * We probably want to make the DataSource serializable as well, so that we can share
//!   source-level state with workers, separately from partition serialization.
//! * We should add a way for the client to negotiate capabilities with the data source, for
//!   example which encodings it knows about.

pub mod plan;
pub mod row_mask;
pub mod scheduler;
pub mod segments;
pub mod selection;
pub mod task;

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
pub use scheduler::ScanMeta;
pub use scheduler::ScanScheduler;
pub use scheduler::ScanSchedulerConfig;
pub use scheduler::ScanSchedulerProvider;
pub use scheduler::ScanSchedulerSession;
pub use scheduler::ScanSchedulerSessionExt;
pub use scheduler::ScanTicket;
pub use scheduler::ScanWorkClass;
pub use scheduler::WorkPermit;
pub use scheduler::WorkRequest;
use selection::Selection;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::root::Root;
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

    /// Whether this source can split one scan into stable round-robin morsel partitions.
    ///
    /// Engines can use this to expose parallel scan partitions even when the underlying
    /// source does not have enough file-level partitions. This is an execution hint, not
    /// a guarantee that each partition has contiguous rows.
    fn supports_morsel_partitioning(&self) -> bool {
        false
    }

    /// Plans one scan into round-robin morsel partitions.
    ///
    /// Implementations should return `Ok(None)` when they cannot share one planned scan across
    /// output partitions. When supported, the returned plan should cap its actual partition count
    /// at the number of planned morsels and each partition should share the same planned scan
    /// state instead of re-planning independently.
    async fn plan_morsel_partitions(
        &self,
        _scan_request: ScanRequest,
        _target_partitions: usize,
    ) -> VortexResult<Option<PlannedMorselScanRef>> {
        Ok(None)
    }

    /// Returns a scan over the source.
    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef>;

    /// Returns metadata aggregate statistics for `expr` over the unfiltered source.
    ///
    /// The returned vector is positional against `funcs`. Each value is exact, inexact, or absent
    /// for the requested aggregate function. The default implementation bridges simple top-level
    /// field expressions through [`Self::field_statistics`] for compatibility with older sources.
    async fn statistics(
        &self,
        expr: &Expression,
        funcs: &[AggregateFnRef],
    ) -> VortexResult<Vec<Precision<Scalar>>> {
        let Some(field_path) = root_field_path(expr) else {
            return Ok(absent_statistics(funcs));
        };
        let Some(field_dtype) = field_path.resolve(self.dtype().clone()) else {
            return Ok(absent_statistics(funcs));
        };
        let stats = self.field_statistics(&field_path).await?;

        funcs
            .iter()
            .map(|func| {
                let Some(stat) = Stat::from_aggregate_fn(func) else {
                    return Ok(Precision::Absent);
                };
                let Some(dtype) = stat.dtype(&field_dtype) else {
                    return Ok(Precision::Absent);
                };
                Ok(stats.get(stat).into_scalar(dtype))
            })
            .collect()
    }

    /// Returns the statistics for a given field.
    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet>;
}

fn absent_statistics(funcs: &[AggregateFnRef]) -> Vec<Precision<Scalar>> {
    funcs.iter().map(|_| Precision::Absent).collect()
}

fn root_field_path(expr: &Expression) -> Option<FieldPath> {
    if expr.is::<Root>() {
        return Some(FieldPath::root());
    }
    let field = expr.as_opt::<GetItem>()?;
    expr.child(0)
        .is::<Root>()
        .then(|| FieldPath::from_name(field.clone()))
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
    /// Optional limit on the number of rows returned by scan. Limits are applied after all
    /// filtering and row selection.
    pub limit: Option<u64>,
    /// Optional scheduler provider override for this scan.
    ///
    /// When absent, a data source should use the provider configured on its [`VortexSession`].
    pub scheduler_provider: Option<Arc<ScanSchedulerProvider>>,
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
            scheduler_provider: None,
        }
    }
}

/// A boxed data source scan.
pub type DataSourceScanRef = Box<dyn DataSourceScan>;

/// A reference-counted scan that has already been planned into morsel partitions.
pub type PlannedMorselScanRef = Arc<dyn PlannedMorselScan>;

/// A planned scan that can return shared execution partitions.
pub trait PlannedMorselScan: 'static + Send + Sync {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// The exact number of non-empty execution partitions in this plan.
    fn partition_count(&self) -> usize;

    /// Returns one planned execution partition.
    fn partition(self: Arc<Self>, partition: usize) -> VortexResult<PartitionRef>;
}

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
