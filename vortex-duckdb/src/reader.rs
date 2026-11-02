// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Custom I/O services for DuckDB's V2 Vortex scans.

use std::sync::Arc;

use parking_lot::RwLock;
use url::Url;
use vortex::error::VortexResult;
use vortex::io::request::IoService;

/// Supplies a file's V2 I/O service. Compute continues on DuckDB's scan threads.
///
/// Services must read the same immutable file as the URL. They must support scopes driven by
/// different DuckDB workers and wake a waiting worker
/// when a read completes. Returning `None` preserves the file's existing I/O service.
pub trait VortexScanIoFactory: Send + Sync {
    /// Opens a service for this file, or declines to override its reader.
    fn create_io_service(&self, url: &Url) -> VortexResult<Option<Arc<dyn IoService>>>;
}

static FACTORY: RwLock<Option<Arc<dyn VortexScanIoFactory>>> = RwLock::new(None);

/// Selects I/O for V2 files opened from now on. V1 scans keep their existing reader.
///
/// This setting is process-wide, like [`crate::set_scan_v2`]. Each opened file retains a snapshot
/// of the factory, so changing it does not replace the service of a scan already in progress.
pub fn set_scan_io_factory(factory: Option<Arc<dyn VortexScanIoFactory>>) {
    *FACTORY.write() = factory;
}

pub(crate) fn scan_io_factory() -> Option<Arc<dyn VortexScanIoFactory>> {
    FACTORY.read().clone()
}
