// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Multi-file data source for scanning across multiple Vortex files.
//!
//! [`MultiFileDataSource`] discovers and opens multiple Vortex files from a glob pattern,
//! presenting them as a single [`DataSource`] for scanning. Footer caching is handled
//! automatically via the session's [`MultiFileSession`].
//!
//! Use [`MultiFileDataSource::builder`] to construct a data source:
//!
//! ```ignore
//! let ds = MultiFileDataSource::builder(session)
//!     .with_glob_url("/data/*.vortex")
//!     .build()
//!     .await?;
//! ```

mod builder;
pub mod session;

pub use builder::MultiFileDataSourceBuilder;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scan::api::DataSource;
use vortex_scan::api::DataSourceScanRef;
use vortex_scan::api::Estimate;
use vortex_scan::api::ScanRequest;
use vortex_scan::api::SplitRef;
use vortex_scan::multi::MultiDataSource;
use vortex_session::VortexSession;

/// A [`DataSource`] that scans across multiple Vortex files, presenting them as a single source.
///
/// Constructed via [`MultiFileDataSource::builder`]. Internally delegates to
/// [`MultiDataSource`] for scan orchestration, prefetching, and split interleaving.
pub struct MultiFileDataSource {
    inner: MultiDataSource,
    file_count: usize,
}

impl MultiFileDataSource {
    /// Create a new builder for a multi-file data source.
    pub fn builder(session: VortexSession) -> MultiFileDataSourceBuilder {
        MultiFileDataSourceBuilder::new(session)
    }

    /// Returns the number of files in this data source.
    pub fn file_count(&self) -> usize {
        self.file_count
    }
}

impl DataSource for MultiFileDataSource {
    fn dtype(&self) -> &DType {
        self.inner.dtype()
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
