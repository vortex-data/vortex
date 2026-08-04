// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use itertools::Itertools;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::cloud::Registry;
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
use vortex_utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::SESSION;
use crate::duckdb::BindInputRef;
use crate::duckdb::ExtractedValue;

/// Process-wide registry, so repeated scans against the same bucket share one client.
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

fn resolve_filesystem(base_url: &Url) -> VortexResult<FileSystemRef> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if base_url.scheme() == "file" {
        return Ok(Arc::new(ObjectStoreFileSystem::local(RUNTIME.handle())));
    }

    // `base_url` has its path cleared by the caller, so the resolved path is empty and only the
    // store matters here. Going through the shared registry means DuckDB resolves the same set of
    // schemes as the Python and Java bindings, including the OpenDAL-backed ones when the
    // `opendal` feature is on.
    let (object_store, _) = REGISTRY.resolve(base_url)?;

    Ok(Arc::new(ObjectStoreFileSystem::new(
        Arc::new(Compat::new(object_store)),
        RUNTIME.handle(),
    )))
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

    // Cache filesystems by base URL to avoid resolving the same filesystem multiple times.
    let mut fs_cache: HashMap<Url, FileSystemRef> = HashMap::new();
    for glob_url in &glob_urls {
        let mut base_url = glob_url.clone();
        base_url.set_path("");
        if !fs_cache.contains_key(&base_url) {
            let fs = resolve_filesystem(&base_url)?;
            fs_cache.insert(base_url, fs);
        }
    }

    RUNTIME.block_on(async {
        let mut builder = MultiFileDataSource::new(SESSION.clone());

        for glob_url in &glob_urls {
            let mut base_url = glob_url.clone();
            base_url.set_path("");
            let fs = fs_cache
                .get(&base_url)
                .map(Arc::clone)
                .unwrap_or_else(|| unreachable!("fs should be cached for all base URLs"));
            builder = builder.with_glob(glob_url.path(), Some(fs));
        }

        builder.build().await
    })
}
