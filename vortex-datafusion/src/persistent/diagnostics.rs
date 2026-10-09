// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in cache ablation for benchmarks over immutable files. This retains a cache per object
//! for the process lifetime and is not a general-purpose, globally bounded application cache.

use std::sync::Arc;
use std::sync::LazyLock;

use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use datafusion_datasource::PartitionedFile;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use tokio::sync::OnceCell;
use vortex::file::VortexFile;
use vortex::io::VortexReadAt;
use vortex::layout::segments::MokaSegmentCache;
use vortex::layout::segments::NoOpSegmentCache;
use vortex::layout::segments::SegmentCache;
use vortex::layout::segments::SegmentEviction;
use vortex::layout::segments::SegmentId;
use vortex_utils::aliases::dash_map::DashMap;

static CAPACITY: LazyLock<Option<String>> =
    LazyLock::new(|| std::env::var("VORTEX_BENCH_SEGMENT_CACHE_MB").ok());
static PRELOAD: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("VORTEX_BENCH_PRELOAD_SEGMENTS").is_ok_and(|value| value == "1")
});
static CACHES: LazyLock<DashMap<CacheKey, BenchmarkCache>> = LazyLock::new(DashMap::default);

#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    uri: Arc<str>,
    size: u64,
    modified: String,
    etag: Option<String>,
    version: Option<String>,
}

#[derive(Clone)]
pub(super) struct BenchmarkCache {
    pub(super) segments: Arc<dyn SegmentCache>,
    preloaded: Arc<OnceCell<()>>,
    capacity: u64,
}

impl BenchmarkCache {
    pub(super) async fn preload(&self, file: &VortexFile) -> DFResult<()> {
        if !*PRELOAD || self.capacity == 0 {
            return Ok(());
        }
        self.preloaded
            .get_or_try_init(|| async {
                let source = file.segment_source();
                let count = u32::try_from(file.footer().segment_map().len())
                    .map_err(|e| exec_datafusion_err!("Too many segments: {e}"))?;
                stream::iter(0..count)
                    .map(|id| source.request(SegmentId::from(id)))
                    .buffer_unordered(64)
                    .try_for_each(|_| async { Ok(()) })
                    .await
                    .map_err(|e| exec_datafusion_err!("Preloading segments failed: {e}"))
            })
            .await?;
        Ok(())
    }
}

pub(super) fn benchmark_segment_cache(
    reader: &dyn VortexReadAt,
    file: &PartitionedFile,
) -> DFResult<Option<BenchmarkCache>> {
    let Some(value) = CAPACITY.as_ref() else {
        return Ok(None);
    };
    let bytes = value
        .parse::<u64>()
        .ok()
        .and_then(|mb| mb.checked_mul(1024 * 1024))
        .ok_or_else(|| exec_datafusion_err!("Invalid VORTEX_BENCH_SEGMENT_CACHE_MB: {value}"))?;
    // Zero measures the segment-source adapter without retaining bytes, controlling for the
    // different routing used by V2's uncached direct range service.
    if bytes == 0 {
        return Ok(Some(BenchmarkCache {
            segments: Arc::new(NoOpSegmentCache),
            preloaded: Arc::default(),
            capacity: 0,
        }));
    }
    if *PRELOAD && bytes < file.object_meta.size {
        return Err(exec_datafusion_err!(
            "Preloaded cache must fit the entire file ({} bytes)",
            file.object_meta.size
        ));
    }
    let uri = reader.uri().ok_or_else(|| {
        exec_datafusion_err!("Benchmark segment caching requires an identified reader")
    })?;
    // Segment ids are file-local. Object metadata also distinguishes file replacements.
    let key = CacheKey {
        uri: Arc::clone(uri),
        size: file.object_meta.size,
        modified: file.object_meta.last_modified.to_rfc3339(),
        etag: file.object_meta.e_tag.clone(),
        version: file.object_meta.version.clone(),
    };
    Ok(Some(
        CACHES
            .entry(key)
            .or_insert_with(|| BenchmarkCache {
                segments: Arc::new(
                    MokaSegmentCache::new(bytes, SegmentEviction::TinyLfu).for_file(""),
                ),
                preloaded: Arc::default(),
                capacity: bytes,
            })
            .value()
            .clone(),
    ))
}
