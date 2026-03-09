// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::absolute;
use std::sync::Arc;

use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::multi::MultiFileDataSource;
use vortex::io::runtime::BlockingRuntime;
use vortex::scan::api::DataSourceRef;

use crate::RUNTIME;
use crate::SESSION;
use crate::datasource::DataSourceTableFunction;
use crate::duckdb::BindInputRef;
use crate::duckdb::ClientContextRef;
use crate::duckdb::LogicalType;
use crate::filesystem::resolve_filesystem;

/// Vortex multi-file scan table function (`vortex_scan` / `read_vortex`).
///
/// Takes a file glob parameter and resolves it into a [`MultiFileDataSource`].
/// All other table function logic is provided by the blanket [`DataSourceTableFunction`]
/// implementation.
#[derive(Debug)]
pub struct VortexMultiFileScan;

impl DataSourceTableFunction for VortexMultiFileScan {
    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(ctx: &ClientContextRef, input: &BindInputRef) -> VortexResult<DataSourceRef> {
        let glob_url_parameter = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        // Parse the URL and separate the base URL (keep scheme, host, etc.) from the path.
        let glob_url_str = glob_url_parameter.as_string();

        let glob_url = Url::parse(&glob_url_str).or_else(|_| {
            let glob_url = glob_url_str.as_str();
            let path = absolute(Path::new(glob_url));
            let path = path.map_err(|e| vortex_err!("Failed making {glob_url} absolute: {e}"))?;
            Url::from_file_path(path).map_err(|_| vortex_err!("Neither URL nor path: {glob_url}"))
        })?;

        let mut base_url = glob_url.clone();
        base_url.set_path("");

        let fs = resolve_filesystem(&base_url, ctx)?;

        RUNTIME.block_on(async {
            let builder = MultiFileDataSource::new(SESSION.clone())
                .with_filesystem(fs)
                .with_glob(glob_url.path());
            let ds = builder.build().await?;
            VortexResult::Ok(Arc::new(ds) as DataSourceRef)
        })
    }
}
