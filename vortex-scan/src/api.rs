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
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

/// Create a Vortex source from serialized configuration.
///
/// Providers can be registered with Vortex under a specific
#[async_trait(?Send)]
pub trait DataSourceProvider: 'static {
    /// Attempt to initialize a new source.
    ///
    /// Returns `Ok(None)` if the provider cannot handle the given URI.
    async fn initialize(
        &self,
        uri: String,
        session: &VortexSession,
    ) -> VortexResult<Option<DataSourceRef>>;
}

/// A reference-counted data source.
pub type DataSourceRef = Arc<dyn DataSource>;

/// A data source represents a streamable dataset that can be scanned with projection and filter
/// expressions. Each scan produces splits that can be executed (potentially in parallel) to read
/// data. Each split can be serialized for remote execution.
///
/// The DataSource may be used multiple times to create multiple scans, whereas each scan and each
/// split of a scan can only be consumed once.
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the source.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Returns a scan over the source.
    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef>;

    /// Serialize a split from this data source.
    fn serialize_split(&self, split: &dyn Split) -> VortexResult<Vec<u8>>;

    /// Deserialize a split that was previously serialized from a compatible data source.
    fn deserialize_split(&self, data: &[u8]) -> VortexResult<SplitRef>;
}

/// A request to scan a data source.
#[derive(Debug, Clone, Default)]
pub struct ScanRequest {
    /// Projection expression, `None` implies `root()`.
    pub projection: Option<Expression>,
    /// Filter expression, `None` implies no filter.
    pub filter: Option<Expression>,
    /// Optional limit on the number of rows to scan.
    pub limit: Option<u64>,
}

/// A boxed data source scan.
pub type DataSourceScanRef = Box<dyn DataSourceScan>;

/// A data source scan produces splits that can be executed to read data from the source.
#[async_trait]
pub trait DataSourceScan: 'static + Send {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// An estimate of the remaining splits.
    fn remaining_splits_estimate(&self) -> Estimate<usize>;

    /// Returns the next batch of splits to be processed.
    ///
    /// This should not return _more_ than `max_splits` splits, but may return fewer.
    async fn next_splits(&mut self, max_splits: usize) -> VortexResult<Vec<SplitRef>>;
}

/// A stream of splits.
pub type SplitStream = BoxStream<'static, VortexResult<SplitRef>>;

/// A reference-counted split.
pub type SplitRef = Box<dyn Split>;

/// A split represents a unit of work that can be executed to produce a stream of arrays.
pub trait Split: 'static + Send {
    /// Downcast the split to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Executes the split.
    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream>;

    /// Returns an estimate of the row count for this split.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Returns an estimate of the byte size for this split.
    fn byte_size_estimate(&self) -> Estimate<u64>;
}

/// An estimate that can be exact, an upper bound, or unknown.
#[derive(Default)]
pub enum Estimate<T> {
    /// The exact value.
    Exact(T),
    /// An upper bound on the value.
    UpperBound(T),
    /// The value is unknown.
    #[default]
    Unknown,
}
