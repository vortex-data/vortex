// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exercise cache lookup ordering with the file source's real physical-read coalescer.

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_file::SegmentSpec;
use vortex_file::segments::FileSegmentSource;
use vortex_file::segments::RequestMetrics;
use vortex_io::CoalesceConfig;
use vortex_io::ReadAtRequest;
use vortex_io::VortexReadAt;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_layout::segments::MokaSegmentCache;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheSourceAdapter;
use vortex_layout::segments::SegmentEviction;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::DefaultMetricsRegistry;

const SEGMENT_LENGTH: u32 = 4096;
const SEGMENT_BYTES: usize = SEGMENT_LENGTH as usize;
const SEGMENTS: u32 = 16;

#[derive(Clone)]
struct CountingReadAt {
    ranges: Arc<Mutex<Vec<ReadAtRequest>>>,
    latency: Duration,
}

impl VortexReadAt for CountingReadAt {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::object_storage())
    }

    fn concurrency(&self) -> usize {
        4
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        async { Ok(u64::from(SEGMENTS) * u64::from(SEGMENT_LENGTH)) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.ranges
            .lock()
            .push(ReadAtRequest::new(offset, length, alignment));
        let latency = self.latency;
        async move {
            if !latency.is_zero() {
                tokio::time::sleep(latency).await;
            }
            Ok(BufferHandle::new_host(
                ByteBuffer::from(vec![0; length]).aligned(alignment),
            ))
        }
        .boxed()
    }
}

// The pre-fix adapter is the control, compiled into the same benchmark binary so the backend,
// runtime, cache, and coalescer are identical between measurements.
struct RegisterBeforeLookup {
    source: Arc<dyn SegmentSource>,
    cache: Arc<dyn SegmentCache>,
}

impl SegmentSource for RegisterBeforeLookup {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let delegate = self.source.request(id);
        let cache = Arc::clone(&self.cache);
        async move {
            if let Ok(Some(buffer)) = cache.get(id).await {
                return Ok(BufferHandle::new_host(buffer));
            }
            let buffer = delegate.await?;
            if let Some(buffer) = buffer.as_host_opt() {
                drop(cache.put(id, buffer.clone()).await);
            }
            Ok(buffer)
        }
        .boxed()
    }
}

#[derive(Clone, Copy, Debug)]
enum Demand {
    Concurrent,
    Ordered,
}

#[derive(Debug)]
struct Measurement {
    ranges: Vec<ReadAtRequest>,
    elapsed: Duration,
}

async fn measure(
    control: bool,
    cached_suffix: u32,
    demand: Demand,
    latency: Duration,
) -> VortexResult<Measurement> {
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let registry = DefaultMetricsRegistry::default();
    let segments: Arc<[SegmentSpec]> = (0..SEGMENTS)
        .map(|id| SegmentSpec {
            offset: u64::from(id) * u64::from(SEGMENT_LENGTH),
            length: SEGMENT_LENGTH,
            alignment: Alignment::none(),
        })
        .collect();
    let source: Arc<dyn SegmentSource> = Arc::new(FileSegmentSource::open(
        segments,
        CountingReadAt {
            ranges: Arc::clone(&ranges),
            latency,
        },
        TokioRuntime::current(),
        RequestMetrics::new(&registry, vec![]),
    ));
    let cache = MokaSegmentCache::new(1 << 20, SegmentEviction::Lru);
    let file_cache: Arc<dyn SegmentCache> = Arc::new(cache.for_file("benchmark-file"));
    for id in SEGMENTS - cached_suffix..SEGMENTS {
        file_cache
            .put(
                SegmentId::from(id),
                ByteBuffer::from(vec![0; SEGMENT_BYTES]),
            )
            .await?;
    }
    cache.run_pending_tasks().await;
    let source: Arc<dyn SegmentSource> = if control {
        Arc::new(RegisterBeforeLookup {
            source,
            cache: file_cache,
        })
    } else {
        Arc::new(SegmentCacheSourceAdapter::new(file_cache, source))
    };
    let start = Instant::now();
    let requests = (0..SEGMENTS)
        .map(|id| source.request(SegmentId::from(id)))
        .collect::<Vec<_>>();
    match demand {
        Demand::Concurrent => {
            for result in futures::future::join_all(requests).await {
                assert_eq!(result?.len(), SEGMENT_BYTES);
            }
        }
        Demand::Ordered => {
            for request in requests {
                assert_eq!(request.await?.len(), SEGMENT_BYTES);
            }
        }
    }
    let elapsed = start.elapsed();
    let ranges = ranges.lock().clone();
    Ok(Measurement { ranges, elapsed })
}

#[tokio::test]
async fn cold_misses_preserve_physical_read_coalescing() -> VortexResult<()> {
    for demand in [Demand::Concurrent, Demand::Ordered] {
        let control = measure(true, 0, demand, Duration::ZERO).await?;
        let candidate = measure(false, 0, demand, Duration::ZERO).await?;
        assert_eq!(control.ranges, candidate.ranges);
        assert_eq!(candidate.ranges.len(), 1);
        assert_eq!(
            candidate.ranges[0].length,
            SEGMENTS as usize * SEGMENT_BYTES
        );
    }
    Ok(())
}

#[tokio::test]
async fn cached_suffix_is_excluded_from_coalesced_reads() -> VortexResult<()> {
    for demand in [Demand::Concurrent, Demand::Ordered] {
        let control = measure(true, 8, demand, Duration::ZERO).await?;
        let candidate = measure(false, 8, demand, Duration::ZERO).await?;
        assert_eq!(control.ranges.len(), 1);
        let control_segments = match demand {
            Demand::Concurrent => 8,
            Demand::Ordered => SEGMENTS as usize,
        };
        assert_eq!(control.ranges[0].length, control_segments * SEGMENT_BYTES);
        assert_eq!(candidate.ranges.len(), 1);
        assert_eq!(candidate.ranges[0].offset, 0);
        assert_eq!(candidate.ranges[0].length, 8 * SEGMENT_BYTES);
    }
    Ok(())
}

#[tokio::test]
#[ignore = "release-mode coalescing benchmark; run explicitly with --ignored --nocapture"]
async fn benchmark_cache_lookup_coalescing() -> VortexResult<()> {
    // A fixed simulated backend RTT makes loss of coalescing visible without network noise.
    // Ordered demand intentionally does not poll unused futures just to improve the result.
    for (latency, repetitions) in [(Duration::ZERO, 101), (Duration::from_millis(2), 9)] {
        for cached_suffix in [0, 8, 16] {
            for demand in [Demand::Concurrent, Demand::Ordered] {
                for control in [true, false] {
                    let mut elapsed = Vec::new();
                    let mut counts = Vec::new();
                    let mut bytes = Vec::new();
                    for _ in 0..repetitions {
                        let measurement = measure(control, cached_suffix, demand, latency).await?;
                        elapsed.push(measurement.elapsed.as_secs_f64() * 1000.0);
                        counts.push(measurement.ranges.len());
                        bytes.push(measurement.ranges.iter().map(|r| r.length).sum::<usize>());
                    }
                    elapsed.sort_by(f64::total_cmp);
                    counts.sort_unstable();
                    bytes.sort_unstable();
                    let demand_name = match demand {
                        Demand::Concurrent => "concurrent",
                        Demand::Ordered => "ordered",
                    };
                    println!(
                        "adapter={} latency_us={} repetitions={repetitions} cached_suffix={cached_suffix} demand={demand_name} physical_reads={}..{} bytes={}..{} median_ms={:.6}",
                        if control { "before" } else { "after" },
                        latency.as_micros(),
                        counts[0],
                        counts[counts.len() - 1],
                        bytes[0],
                        bytes[bytes.len() - 1],
                        elapsed[elapsed.len() / 2],
                    );
                }
            }
        }
    }
    Ok(())
}
