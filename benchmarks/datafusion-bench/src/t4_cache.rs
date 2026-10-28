// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! t4 in front of the benchmark's object store, when `--opt t4-policy=...` asks for it: t4, a
//! cache of the store in memory and on local drives, serves the session's lists and every Vortex
//! file's reads ([`T4ReadAt`]).
//!
//! - `t4-policy`: `clock`, t4's defaults; or every fetched block kept in `memory`, on `disk`, or
//!   `none` kept, so that a run measures one tier alone.
//! - `t4-memory-gib`: the memory tier.
//! - `t4-disk`: the cache files, each `path:GiB`, comma separated.
//! - `t4-coalesce`: `on`, Vortex merging nearby reads as it does for object stores, the default;
//!   or `off`, t4 taking each read as asked.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::prelude::SessionContext;
use datafusion_common::Result as DFResult;
use object_store::ObjectStore;
use t4::T4Caching;
use t4::policy::Admit;
use t4::policy::FixedPlacementPolicy;
use url::Url;
use vortex::array::memory::MemorySessionExt;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::t4::T4ReadAt;
use vortex::session::VortexSession;
use vortex_bench::Opts;
use vortex_datafusion::reader::VortexReaderFactory;

const GIB: u64 = 1 << 30;

/// Puts t4 in front of `store`, which serves `data_url`, if `opts` ask for it: t4 is registered
/// as the session's store there, and the factory returned reads every Vortex file through it.
pub fn t4_in_front(
    session: &SessionContext,
    opts: &Opts,
    data_url: &Url,
    store: Arc<dyn ObjectStore>,
) -> anyhow::Result<Option<Arc<dyn VortexReaderFactory>>> {
    let Some(policy) = opts.get("t4-policy") else {
        return Ok(None);
    };
    anyhow::ensure!(
        matches!(data_url.scheme(), "s3" | "gs"),
        "t4 caches an object store: read the data from S3 or GCS (--opt remote-data-dir=...)"
    );
    let memory_gib: u64 = opts
        .get_as("t4-memory-gib")
        .context("t4 needs --opt t4-memory-gib=<GiB>")?;
    let disk_files = opts
        .get("t4-disk")
        .context("t4 needs --opt t4-disk=<path>:<GiB>[,...]")?;

    let mut builder = T4Caching::builder(store)
        .with_memory_capacity(memory_gib * GIB)
        .with_sleep(|after| Box::pin(tokio::time::sleep(after)));
    for disk_file in disk_files.split(',') {
        let (path, gib) = disk_file
            .rsplit_once(':')
            .context("each t4 cache file is <path>:<GiB>")?;
        builder = builder.with_disk_file(PathBuf::from(path), gib.parse::<u64>()? * GIB);
    }
    let builder = match policy {
        "clock" => builder,
        "memory" => builder
            .with_placement_policy(|sizes| FixedPlacementPolicy::new(sizes, Admit::Memory)),
        "disk" => {
            builder.with_placement_policy(|sizes| FixedPlacementPolicy::new(sizes, Admit::Disk))
        }
        "none" => builder
            .with_placement_policy(|sizes| FixedPlacementPolicy::new(sizes, Admit::Discard)),
        other => anyhow::bail!("t4-policy is clock, memory, disk or none, not {other}"),
    };
    let coalesce_config = match opts.get("t4-coalesce").unwrap_or("on") {
        "on" => Some(CoalesceConfig::object_storage()),
        "off" => None,
        other => anyhow::bail!("t4-coalesce is on or off, not {other}"),
    };
    let cache = builder.build()?;

    let bucket = &data_url[url::Position::BeforeHost..url::Position::AfterHost];
    let root = Url::parse(&format!("{}://{bucket}/", data_url.scheme()))?;
    session.register_object_store(&root, Arc::new(cache.clone()));

    Ok(Some(Arc::new(T4ReaderFactory {
        cache,
        coalesce_config,
    })))
}

/// Opens each file DataFusion lists through t4, with the size and e_tag the list gave, so that
/// opening asks the store nothing.
#[derive(Debug)]
struct T4ReaderFactory {
    cache: T4Caching,
    coalesce_config: Option<CoalesceConfig>,
}

impl VortexReaderFactory for T4ReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        let file = self.cache.open_listed(file.object_meta.clone());
        Ok(Arc::new(
            T4ReadAt::new_with_allocator(file, session.handle(), session.allocator())
                .with_coalesce_config(self.coalesce_config),
        ))
    }
}
