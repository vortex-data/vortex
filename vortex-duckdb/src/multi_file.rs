// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::multi::MultiFileDataSource;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::multi::MultiLayoutDataSource;

use crate::REGISTRY;
use crate::RUNTIME;
use crate::SESSION;
use crate::duckdb::BindInputRef;
use crate::duckdb::ExtractedValue;

fn resolve_filesystem(glob_url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if glob_url.scheme() == "file" {
        return Ok((
            Arc::new(ObjectStoreFileSystem::local(RUNTIME.handle())),
            glob_url.path().to_string(),
        ));
    }

    // The full URL goes through the shared registry, which reports the glob as a path *within*
    // the store it returns. For most schemes the store is mounted at the URL authority, so the
    // path is the whole URL path — but not for all of them: an `hf://` store is rooted at a
    // repository and revision, which occupy path segments. Only the registry knows how deep the
    // store is mounted, so globbing anything other than the path it reports would address the
    // wrong keys. Going through the registry also means DuckDB resolves the same set of schemes
    // as the Python and Java bindings, including the OpenDAL-backed ones when the `opendal`
    // feature is on. The registry caches one client per store prefix, so repeated scans against
    // the same bucket or repository share a client even though the filesystem wrapper is rebuilt.
    let (object_store, path) = REGISTRY.resolve(glob_url)?;

    Ok((
        Arc::new(ObjectStoreFileSystem::new(
            Arc::new(Compat::new(object_store)),
            RUNTIME.handle(),
        )),
        path.to_string(),
    ))
}

/// Shared bind logic for both single-glob and multi-glob variants.
pub fn bind_multi_file_scan(input: &BindInputRef) -> VortexResult<MultiLayoutDataSource> {
    let glob_url_parameter = input
        .get_parameter(0)
        .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

    // The input to the table function can either be a single glob, or a List of glob patterns.
    let glob_strings: Vec<String> = match glob_url_parameter.extract() {
        ExtractedValue::Varchar(glob) => {
            vec![glob.to_string()]
        }
        ExtractedValue::List(globs) => globs
            .into_iter()
            .map(|glob| {
                let ExtractedValue::Varchar(string) = glob.extract() else {
                    vortex_bail!("list element must be Varchar type")
                };

                Ok(string.to_string())
            })
            .try_collect()?,
        _ => vortex_bail!("Invalid argument to read_vortex table function"),
    };

    // Parse each glob URL and resolve its filesystem.
    let mut glob_urls: Vec<Url> = Vec::with_capacity(glob_strings.len());
    for glob_str in &glob_strings {
        glob_urls.push(parse_uri_or_path(glob_str)?);
    }

    let resolved = glob_urls
        .iter()
        .map(resolve_filesystem)
        .collect::<VortexResult<Vec<_>>>()?;

    RUNTIME.block_on(async {
        let mut builder = MultiFileDataSource::new(SESSION.clone());

        for (fs, glob) in resolved {
            builder = builder.with_glob(&glob, Some(fs));
        }

        builder.build().await
    })
}
