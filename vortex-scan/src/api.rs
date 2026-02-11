// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! The Vortex Scan API implements an abstract table scan interface that can be used to
//! read data from various data sources.
//!
//! It supports arbitrary projection expressions, filter expressions, and limit pushdown as well
//! as mechanisms for parallel and distributed execution via splits.
//!
//! The API is currently under development and may change in future releases, however we hope to
//! stabilize into stable C ABI for use within foreign language bindings.
//!
//! ## Open Issues
//!
//! * We probably want to make the DataSource serializable as well, so that we can share
//!   source-level state with workers, separately from split serialization.
//! * We should add a way for the client to negotiate capabilities with the data source, for
//!   example which encodings it knows about.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Selection;

/// A sendable stream of splits.
pub type SplitStream = BoxStream<'static, VortexResult<SplitRef>>;

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
/// expressions. Each scan produces splits that can be executed in parallel to read data. Each
/// split can be serialized for remote execution.
///
/// The DataSource may be used multiple times to create multiple scans, whereas each scan and each
/// split of a scan can only be consumed once.
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the source.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Serialize the [`DataSource`] to pass to a remote worker.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Deserialize a split that was previously serialized from a compatible data source.
    fn deserialize_split(&self, data: &[u8], session: &VortexSession) -> VortexResult<SplitRef>;

    /// Returns a scan over the source.
    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef>;
}

/// A request to scan a data source.
#[derive(Debug, Clone, Default)]
pub struct ScanRequest {
    /// Projection expression, `None` implies `root()`.
    pub projection: Option<Expression>,
    /// Filter expression, `None` implies no filter.
    pub filter: Option<Expression>,
    /// The row range to read.
    pub row_range: Option<Range<u64>>,
    /// A row selection to apply to the scan. The selection identifies rows within the specified
    /// row range.
    pub selection: Selection,
    /// Optional limit on the number of rows returned by scan. Limits are applied after all
    /// filtering and row selection.
    pub limit: Option<u64>,
}

/// A boxed data source scan.
pub type DataSourceScanRef = Box<dyn DataSourceScan>;

/// A data source scan produces splits that can be executed to read data from the source.
pub trait DataSourceScan: 'static + Send {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// An estimate of the total number of splits.
    fn splits_estimate(&self) -> Estimate<usize>;

    /// Returns a stream of splits to be processed.
    fn splits(self: Box<Self>) -> SplitStream;
}

/// A reference-counted split.
pub type SplitRef = Box<dyn Split>;

/// A split represents a unit of work that can be executed to produce a stream of arrays.
pub trait Split: 'static + Send {
    /// Downcast the split to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Returns an estimate of the row count for this split.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Returns an estimate of the byte size for this split.
    fn byte_size_estimate(&self) -> Estimate<u64>;

    /// Serialize this split for a remote worker.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Executes the split.
    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream>;
}

/// An estimate that can be exact, an upper bound, or unknown.
#[derive(Clone, Debug)]
pub struct Estimate<T> {
    /// The lower bound
    pub lower: T,
    /// The upper bound
    pub upper: Option<T>,
}

impl<T: Default> Default for Estimate<T> {
    fn default() -> Self {
        Self {
            lower: T::default(),
            upper: None,
        }
    }
}

impl<T: Copy> Estimate<T> {
    /// Creates an exact estimate.
    pub fn exact(value: T) -> Self {
        Self {
            lower: value,
            upper: Some(value),
        }
    }
}
