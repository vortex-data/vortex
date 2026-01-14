// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// Create a Vortex source from serialized configuration.
///
/// Providers can be registered with Vortex under a specific
#[async_trait(?Send)]
pub trait DataSourceProvider: 'static {
    /// URI schemes handled by this source provider.
    ///
    /// TODO(ngates): this might not be the right way to plugin sources.
    fn schemes(&self) -> &[&str];

    /// Initialize a new source.
    async fn init_source(&self, uri: String) -> VortexResult<DataSourceRef>;

    /// Serialize a source split to bytes.
    async fn serialize_split(&self, split: &dyn Split) -> VortexResult<Vec<u8>>;

    /// Deserialize a source split from bytes.
    async fn deserialize_split(&self, data: &[u8]) -> VortexResult<SplitRef>;
}

/// A reference-counted source.
pub type DataSourceRef = Arc<dyn DataSource>;

/// A source represents a streamable dataset that can be scanned with projection and filter
/// expressions. Each scan produces splits that can be executed in parallel to read data.
/// Each split can be serialized for remote execution.
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the source.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Returns a scan over the source.
    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef>;
}

#[derive(Debug, Clone, Default)]
pub struct ScanRequest {
    pub projection: Option<Expression>,
    pub filter: Option<Expression>,
    pub limit: Option<u64>,
}

pub type DataSourceScanRef = Box<dyn DataSourceScan>;

#[async_trait]
pub trait DataSourceScan: 'static + Send + Sync {
    /// The returned dtype of the scan.
    fn dtype(&self) -> &DType;

    /// An estimate of the remaining splits.
    fn remaining_splits_estimate(&self) -> Estimate<usize>;

    /// Returns the next batch of splits to be processed.
    ///
    /// This should not return _more_ than the max_batch_size splits, but may return fewer.
    async fn next_splits(&mut self, max_splits: usize) -> VortexResult<Vec<SplitRef>>;
}

pub type SplitStream = BoxStream<'static, VortexResult<SplitRef>>;
pub type SplitRef = Arc<dyn Split>;

pub trait Split: 'static + Send + Sync {
    /// Downcast the split to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Executes the split.
    fn execute(&self) -> VortexResult<SendableArrayStream>;

    /// Returns an estimate of the row count for this split.
    fn row_count_estimate(&self) -> Estimate<u64>;

    /// Returns an estimate of the byte size for this split.
    fn byte_size_estimate(&self) -> Estimate<u64>;
}

#[derive(Default)]
pub enum Estimate<T> {
    Exact(T),
    UpperBound(T),
    #[default]
    Unknown,
}
