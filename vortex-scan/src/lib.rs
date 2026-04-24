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

pub mod row_mask;
pub mod selection;

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use selection::Selection;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
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
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the un-filtered source.
    fn row_count(&self) -> Option<Precision<u64>> {
        None
    }

    /// Returns an estimate of the byte size of the un-filtered source.
    fn byte_size(&self) -> Option<Precision<u64>> {
        None
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
}

/// A request to scan a data source.
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Projection expression. Defaults to `root()` which returns all columns.
    pub projection: Expression,
    /// Filter expression, `None` implies no filter.
    pub filter: Option<Expression>,
    /// The row range to read.
    pub row_range: Option<Range<u64>>,
    /// A row selection to apply to the scan. The selection identifies rows within the specified
    /// row range.
    pub selection: Selection,
    /// Whether the scan should preserve row order. If false, the scan may produce rows in any
    /// order, for example to enable parallel execution across partitions.
    pub ordered: bool,
    /// Optional limit on the number of rows returned by scan. Limits are applied after all
    /// filtering and row selection.
    pub limit: Option<u64>,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            projection: root(),
            filter: None,
            row_range: None,
            selection: Selection::default(),
            ordered: false,
            limit: None,
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
    fn partition_count(&self) -> Option<Precision<usize>>;

    /// Returns a stream of partitions to be processed.
    fn partitions(self: Box<Self>) -> PartitionStream;
}

/// A reference-counted partition.
pub type PartitionRef = Box<dyn Partition>;

/// A partition represents a unit of work that can be executed to produce a stream of arrays.
pub trait Partition: 'static + Send {
    /// Downcast the partition to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Returns an estimate of the row count for this partition.
    fn row_count(&self) -> Option<Precision<u64>>;

    /// Returns an estimate of the byte size for this partition.
    fn byte_size(&self) -> Option<Precision<u64>>;

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
