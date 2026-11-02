// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Installs the shared completion backend without changing DuckDB's compute scheduling.

use std::sync::Arc;

use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::read_at2::ReadAt2IoService;
use vortex::io::request::IoService;
use vortex_bench::uring::LocalUring;
use vortex_duckdb::reader::VortexScanIoFactory;
use vortex_duckdb::reader::set_scan_io_factory;

struct DuckDbUring(Arc<LocalUring>);

impl VortexScanIoFactory for DuckDbUring {
    fn create_io_service(&self, url: &url::Url) -> VortexResult<Option<Arc<dyn IoService>>> {
        if url.scheme() != "file" {
            return Ok(None);
        }
        let path = url
            .to_file_path()
            .map_err(|()| vortex_err!("Invalid file URL {url}"))?;
        let reader = self.0.reader(&path).map_err(VortexError::external)?;
        Ok(Some(Arc::new(ReadAt2IoService::new(reader))))
    }
}

pub fn configure(scan_v2: bool) -> anyhow::Result<Option<Arc<LocalUring>>> {
    let backend = std::env::var("VORTEX_IO_BACKEND").unwrap_or_default();
    if !scan_v2 || backend.is_empty() || backend == "legacy" {
        return Ok(None);
    }
    anyhow::ensure!(
        backend.starts_with("shared-"),
        "DuckDB completion I/O requires a shared backend; compute runs on DuckDB workers"
    );
    let io = Arc::new(LocalUring::from_env(&backend)?);
    set_scan_io_factory(Some(Arc::new(DuckDbUring(Arc::clone(&io)))));
    Ok(Some(io))
}
