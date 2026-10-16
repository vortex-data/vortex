// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diagnose selection scans on an existing Vortex file.
//!
//! Usage: `selection_scan FILE ITERATIONS ROWS uniform|clustered|all cached|reopen memory|disk
//! time|metrics|check|decode|decode-metrics [SEED]`. Set `VORTEX_SCAN_V2=0/1` in separate processes.
//! Timings exclude instrumentation in `time` and `decode` modes. The flag selects the explicit V2
//! builder here only. `check` compares against a full V1 scan followed by a take.
//! `decode` consumes the selected rows in all columns as Arrow record batches; `decode-metrics`
//! measures allocations while doing the same work.
//! Build with `VORTEX_SELECTION_METRICS=1` for the separate allocation-instrumented executable.

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::hint::black_box;
use std::path::Path;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use futures::StreamExt;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::v2::ScanBuilder as V2ScanBuilder;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

const TRACK_ALLOCATIONS: bool = option_env!("VORTEX_SELECTION_METRICS").is_some();
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

struct Allocator;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn allocated(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size as u64, Ordering::Relaxed);
    let live = LIVE.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: All operations delegate to the same allocator with the original pointer and layout.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let pointer = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if !pointer.is_null() && TRACK_ALLOCATIONS {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let pointer = unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) };
        if !pointer.is_null() && TRACK_ALLOCATIONS {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller guarantees the original allocation and requested size are valid.
        let pointer = unsafe { mimalloc::MiMalloc.realloc(pointer, layout, size) };
        if !pointer.is_null() && TRACK_ALLOCATIONS {
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
            allocated(size);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if TRACK_ALLOCATIONS {
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        // SAFETY: This allocation came from mimalloc with the same layout.
        unsafe { mimalloc::MiMalloc.dealloc(pointer, layout) };
    }
}

struct CountingSegments {
    inner: Arc<dyn SegmentSource>,
    calls: Arc<[AtomicU64]>,
}

impl SegmentSource for CountingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.calls[*id as usize].fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }
}

async fn open(
    session: &VortexSession,
    path: &Path,
    bytes: Option<&ByteBuffer>,
) -> Result<VortexFile> {
    let options = session.open_options().with_layout_reader_cache();
    Ok(match bytes {
        Some(bytes) => options.open_buffer(bytes.clone())?,
        None => options.open_path(path).await?,
    })
}

fn v2() -> bool {
    std::env::var("VORTEX_SCAN_V2").is_ok_and(|value| value == "1")
}

fn v2_scan(
    file: &VortexFile,
    indices: &Buffer<u64>,
    all: bool,
    session: &VortexSession,
) -> Result<V2ScanBuilder> {
    let builder = V2ScanBuilder::new(session.clone(), file.scan_file());
    Ok(if all {
        builder
    } else {
        builder.with_row_indices(StrictSortedBuffer::try_new(indices.clone())?)
    })
}

async fn scan(
    file: &VortexFile,
    indices: &Buffer<u64>,
    all: bool,
    session: &VortexSession,
) -> Result<ArrayRef> {
    let stream = if v2() {
        v2_scan(file, indices, all, session)?.into_stream()?
    } else {
        let builder = file.scan()?;
        let builder = if all {
            builder
        } else {
            builder.with_row_indices(StrictSortedBuffer::try_new(indices.clone())?)
        };
        builder.into_stream()?.boxed()
    };
    let array = ArrayStreamAdapter::new(file.dtype().clone(), stream)
        .read_all()
        .await?;
    Ok(array
        .execute::<Canonical>(&mut session.create_execution_ctx())?
        .into_array())
}

async fn decode(
    file: &VortexFile,
    indices: &Buffer<u64>,
    all: bool,
    session: &VortexSession,
) -> Result<usize> {
    let schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
    let stream = if v2() {
        v2_scan(file, indices, all, session)?
            .into_record_batch_stream(schema)?
            .boxed()
    } else {
        let builder = file.scan()?;
        let builder = if all {
            builder
        } else {
            builder.with_row_indices(StrictSortedBuffer::try_new(indices.clone())?)
        };
        builder.into_record_batch_stream(schema)?.boxed()
    };
    let mut stream = pin!(stream);
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch?.num_rows();
    }
    Ok(rows)
}

fn indices(rows: u64, count: usize, pattern: &str, seed: u64) -> Result<Buffer<u64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(count);
    ensure!(count as u64 <= rows, "Selection exceeds the file row count");
    match pattern {
        "all" => ensure!(count == 0, "Pass ROWS=0 for an all-row scan"),
        "uniform" => {
            while result.len() < count {
                result.push(rng.random_range(0..rows));
                result.sort_unstable();
                result.dedup();
            }
        }
        "clustered" => {
            while result.len() < count {
                let len = (count - result.len()).min(20) as u64;
                let start = rng.random_range(0..=rows - len);
                result.extend(start..start + len);
                result.sort_unstable();
                result.dedup();
            }
        }
        _ => bail!("Unknown pattern {pattern}"),
    }
    Ok(Buffer::from(result))
}

fn rss() -> Result<serde_json::Value> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let get = |name: &str| -> Option<u64> {
        status
            .lines()
            .find(|line| line.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            .map(|kb| kb * 1024)
    };
    Ok(serde_json::json!({"current": get("VmRSS:"), "peak": get("VmHWM:")}))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        matches!(args.len(), 7 | 8),
        "Expected FILE ITERATIONS ROWS uniform|clustered|all cached|reopen memory|disk time|metrics|check|decode|decode-metrics [SEED]"
    );
    let iterations = args[1].parse::<usize>()?;
    ensure!(iterations > 0, "ITERATIONS must be positive");
    let count = args[2].parse::<usize>()?;
    let cached = match args[4].as_str() {
        "cached" => true,
        "reopen" => false,
        other => bail!("Unknown open mode {other}"),
    };
    let metrics = matches!(args[6].as_str(), "metrics" | "decode-metrics");
    ensure!(
        matches!(
            args[6].as_str(),
            "time" | "metrics" | "check" | "decode" | "decode-metrics"
        ),
        "Unknown measurement mode"
    );
    ensure!(
        args[6] == "check" || metrics == TRACK_ALLOCATIONS,
        "Build with VORTEX_SELECTION_METRICS=1 only for metrics mode"
    );
    let session = VortexSession::default().with_tokio();
    let path = Path::new(&args[0]);
    let bytes = match args[5].as_str() {
        "memory" => Some(ByteBuffer::from(std::fs::read(path)?)),
        "disk" => None,
        other => bail!("Unknown storage mode {other}"),
    };
    let file = open(&session, path, bytes.as_ref()).await?;
    let seed = args
        .get(7)
        .map(|seed| seed.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let selected = indices(file.row_count(), count, &args[3], seed)?;
    let all = args[3] == "all";
    let decode_rows = matches!(args[6].as_str(), "decode" | "decode-metrics");
    let expected_rows = if all {
        usize::try_from(file.row_count())?
    } else {
        selected.len()
    };
    if args[6] == "check" {
        let expected = ArrayStreamAdapter::new(
            file.dtype().clone(),
            file.scan()?.prepare()?.execute_stream(None)?,
        )
        .read_all()
        .await?;
        let expected = if all {
            expected
        } else {
            expected.take(selected.clone().into_array())?
        };
        let actual = scan(&file, &selected, all, &session).await?;
        let mut ctx = session.create_execution_ctx();
        let arrow = session.arrow();
        let target = arrow.to_arrow_field("", file.dtype())?;
        let expected = arrow.execute_arrow(expected, Some(&target), &mut ctx)?;
        let actual = arrow.execute_arrow(actual, Some(&target), &mut ctx)?;
        ensure!(
            actual.as_ref() == expected.as_ref(),
            "Selected values differ from full-scan take"
        );
        println!(
            "{}",
            serde_json::json!({"correct": true, "rows": actual.len(), "indices": selected.as_slice()})
        );
        return Ok(());
    }
    let locations = file.scan_file().locations;
    let calls: Arc<[AtomicU64]> = locations.iter().map(|_| AtomicU64::new(0)).collect();
    let instrument = |file: VortexFile| {
        if metrics {
            let source = Arc::new(CountingSegments {
                inner: file.segment_source(),
                calls: Arc::clone(&calls),
            });
            file.with_segment_source(source)
        } else {
            file
        }
    };
    let file = instrument(file);
    let mut runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        for calls in calls.iter() {
            calls.store(0, Ordering::Relaxed);
        }
        let before_live = LIVE.load(Ordering::Relaxed);
        let before_calls = CALLS.load(Ordering::Relaxed);
        let before_bytes = BYTES.load(Ordering::Relaxed);
        PEAK.store(before_live, Ordering::Relaxed);
        let start = Instant::now();
        let reopened = if cached {
            None
        } else {
            Some(instrument(open(&session, path, bytes.as_ref()).await?))
        };
        let input = reopened.as_ref().unwrap_or(&file);
        let (array, actual_rows) = if decode_rows {
            (None, decode(input, &selected, all, &session).await?)
        } else {
            let array = scan(input, &selected, all, &session).await?;
            let rows = array.len();
            (Some(array), rows)
        };
        drop(reopened);
        let elapsed = start.elapsed().as_nanos();
        ensure!(actual_rows == expected_rows, "Wrong selected row count");
        let peak_live = PEAK.load(Ordering::Relaxed).saturating_sub(before_live);
        let output_live = LIVE.load(Ordering::Relaxed) as i128 - before_live as i128;
        drop(black_box(array));
        let retained_live = LIVE.load(Ordering::Relaxed) as i128 - before_live as i128;
        runs.push(serde_json::json!({
            "elapsed_ns": elapsed, "instrumented": metrics,
            "allocation_calls": CALLS.load(Ordering::Relaxed) - before_calls,
            "allocated_bytes": BYTES.load(Ordering::Relaxed) - before_bytes,
            "baseline_live_bytes": before_live,
            "total_peak_live_bytes": before_live + peak_live,
            "post_drop_live_bytes": LIVE.load(Ordering::Relaxed),
            "peak_live_bytes": peak_live, "output_live_bytes": output_live,
            "retained_live_bytes": retained_live,
            "requests": calls.iter().map(|n| n.load(Ordering::Relaxed)).sum::<u64>(),
            "requested_bytes": calls.iter().zip(locations.iter()).map(|(n, l)| n.load(Ordering::Relaxed) * u64::from(l.length)).sum::<u64>(),
            "unique_segments": calls.iter().filter(|n| n.load(Ordering::Relaxed) != 0).count(),
            "unique_bytes": calls.iter().zip(locations.iter()).filter(|(n, _)| n.load(Ordering::Relaxed) != 0).map(|(_, l)| u64::from(l.length)).sum::<u64>(),
        }));
    }
    println!(
        "{}",
        serde_json::json!({"args": args, "indices": selected.as_slice(), "runs": runs, "rss": rss()?, "v2": std::env::var("VORTEX_SCAN_V2").ok()})
    );
    Ok(())
}
