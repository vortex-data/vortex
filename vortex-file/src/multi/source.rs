// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A reusable, engine-agnostic multi-file [`DataSource`] for scanning across multiple Vortex files.
//!
//! [`MultiFileDataSource`] wraps a [`MultiDataSource`] and presents multiple Vortex files as a
//! single scannable data source. It is constructed via [`MultiFileDataSourceBuilder`].
//!
//! # Future Work
//!
//! - **Hive-style partitioning**: Extract partition values from file paths (e.g. `year=2024/month=01/`)
//!   and expose them as virtual columns.
//! - **Virtual columns**: `filename`, `file_row_number`, `file_index`.
//! - **Per-file statistics**: Merge column statistics across files for planner hints.

use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectStore;
use tracing::debug;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scan::api::DataSource;
use vortex_scan::api::DataSourceRef;
use vortex_scan::api::DataSourceScanRef;
use vortex_scan::api::Estimate;
use vortex_scan::api::ScanRequest;
use vortex_scan::api::SplitRef;
use vortex_scan::layout::LayoutReaderDataSource;
use vortex_scan::multi::DataSourceFactory;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::v2::FileStatsLayoutReader;

/// A [`DataSource`] that scans across multiple Vortex files, presenting them as a single source.
///
/// Constructed via [`MultiFileDataSourceBuilder`](super::MultiFileDataSourceBuilder).
/// Internally delegates to [`MultiDataSource`] for scan orchestration, prefetching, and
/// split interleaving.
pub struct MultiFileDataSource {
    dtype: DType,
    inner: MultiDataSource,
    base_url: String,
    file_count: usize,
}

impl MultiFileDataSource {
    pub(super) fn new(
        dtype: DType,
        inner: MultiDataSource,
        base_url: String,
        file_count: usize,
    ) -> Self {
        Self {
            dtype,
            inner,
            base_url,
            file_count,
        }
    }

    /// Returns the base URL prefix for files in this data source.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the number of files in this data source.
    pub fn file_count(&self) -> usize {
        self.file_count
    }
}

impl DataSource for MultiFileDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        self.inner.row_count_estimate()
    }

    fn deserialize_split(&self, data: &[u8], session: &VortexSession) -> VortexResult<SplitRef> {
        self.inner.deserialize_split(data, session)
    }

    fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        self.inner.scan(scan_request)
    }
}

/// A [`DataSourceFactory`] that lazily opens a single Vortex file and wraps it in a
/// [`LayoutReaderDataSource`].
///
/// Handles statistics-based pruning via [`VortexFile::can_prune()`](crate::VortexFile::can_prune).
pub(super) struct VortexFileFactory {
    pub(super) object_store: Arc<dyn ObjectStore>,
    pub(super) path: String,
    pub(super) filter: Option<Expression>,
    pub(super) session: VortexSession,
    pub(super) open_options_fn:
        Arc<dyn Fn(crate::VortexOpenOptions) -> crate::VortexOpenOptions + Send + Sync>,
}

#[async_trait]
impl DataSourceFactory for VortexFileFactory {
    async fn open(&self) -> VortexResult<Option<DataSourceRef>> {
        debug!(path = %self.path, "opening vortex file");
        let options = (self.open_options_fn)(self.session.open_options());
        let file = options
            .open_object_store(&self.object_store, &self.path)
            .await?;

        if let Some(ref filter) = self.filter
            && file.can_prune(filter)?
        {
            debug!(path = %self.path, "pruned file based on statistics");
            return Ok(None);
        }

        let ds = data_source_from_file(&file, &self.session)?;
        debug!(path = %self.path, "opened vortex file");
        Ok(Some(ds))
    }
}

/// Create a [`DataSourceRef`] from a [`VortexFile`], wrapping with
/// [`FileStatsLayoutReader`] when file-level statistics are available.
pub(super) fn data_source_from_file(
    file: &VortexFile,
    session: &VortexSession,
) -> VortexResult<DataSourceRef> {
    let mut reader = file.layout_reader()?;
    if let Some(stats) = file.file_stats().cloned() {
        reader = Arc::new(FileStatsLayoutReader::new(reader, stats, session.clone()));
    }
    Ok(Arc::new(LayoutReaderDataSource::new(
        reader,
        session.clone(),
    )))
}
