// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks many threads awaiting the same segments through a [`SharedSegmentSource`].
//!
//! When a scan cuts a file into parts smaller than a segment, every thread reading one of those
//! parts awaits the same in-flight segment. Each thread here reads every segment in turn, blocking
//! on each, while a background driver completes the reads, so that threads join reads in flight,
//! poll them and are woken by their completion all at once.
//!
//! `file` shares reads from a [`FileSegmentSource`], as a Vortex file does. `future` shares the
//! same reads through a source that only returns their futures, as any other source does.
//!
//! Lock contention does not show up in instruction counts, so CodSpeed does not run this benchmark.

#![cfg_attr(codspeed, allow(unused))]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::LazyLock;
use std::thread::JoinHandle;

use divan::Bencher;
use futures::FutureExt;
use futures::executor::block_on;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::runtime::current::CurrentThreadWorkerPool;
use vortex_io::segments::FileSegmentSource;
use vortex_io::segments::RequestMetrics;
use vortex_io::segments::SegmentFuture;
use vortex_io::segments::SegmentId;
use vortex_io::segments::SegmentLocation;
use vortex_io::segments::SegmentSource;
use vortex_io::segments::SharedSegmentSource;
use vortex_metrics::DefaultMetricsRegistry;

fn main() {
    divan::main();
}

/// Segments every thread reads per iteration.
const SEGMENTS: usize = 1024;

/// Bytes in each segment.
const SEGMENT_LEN: usize = 64;

/// A file whose positional reads complete at once.
#[derive(Clone)]
struct MemoryFile;

impl VortexReadAt for MemoryFile {
    fn concurrency(&self) -> usize {
        64
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        async { Ok((SEGMENTS * SEGMENT_LEN) as u64) }.boxed()
    }

    fn read_at(
        &self,
        _offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async move { Ok(BufferHandle::new_host(ByteBuffer::zeroed(length))) }.boxed()
    }
}

/// Exposes only the futures of the source it wraps.
struct FutureSource(FileSegmentSource);

impl SegmentSource for FutureSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.0.request(id)
    }
}

/// One background thread drives file reads, as a Vortex runtime does.
static RUNTIME: LazyLock<(CurrentThreadRuntime, CurrentThreadWorkerPool)> = LazyLock::new(|| {
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(1);
    (runtime, pool)
});

fn file_source() -> FileSegmentSource {
    let segments = (0..SEGMENTS)
        .map(|segment| SegmentLocation {
            offset: (segment * SEGMENT_LEN) as u64,
            length: SEGMENT_LEN as u32,
            alignment: Alignment::none(),
        })
        .collect::<Arc<[_]>>();
    let metrics = DefaultMetricsRegistry::default();
    FileSegmentSource::open(
        segments,
        MemoryFile,
        RUNTIME.0.handle(),
        RequestMetrics::new(&metrics, vec![]),
    )
}

struct Ready {
    start: Arc<Barrier>,
    readers: Vec<JoinHandle<()>>,
    /// A request for every segment, never polled, so that each segment is read exactly once
    /// however far apart the threads drift.
    _held: Vec<SegmentFuture>,
}

/// Starts `threads` threads that each read every segment from `source` in turn, once released.
fn ready(threads: usize, source: Arc<dyn SegmentSource>) -> Ready {
    let held = (0..SEGMENTS)
        .map(|id| source.request(SegmentId::from(id as u32)))
        .collect();
    let start = Arc::new(Barrier::new(threads + 1));
    let readers = (0..threads)
        .map(|_| {
            let source = Arc::clone(&source);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for id in 0..SEGMENTS {
                    block_on(source.request(SegmentId::from(id as u32))).unwrap();
                }
            })
        })
        .collect();
    Ready {
        start,
        readers,
        _held: held,
    }
}

fn run(ready: Ready) {
    ready.start.wait();
    for reader in ready.readers {
        reader.join().unwrap();
    }
}

#[cfg(not(codspeed))]
#[divan::bench(args = [1, 4, 16], sample_count = 50)]
fn file(bencher: Bencher, threads: usize) {
    bencher
        .with_inputs(|| ready(threads, Arc::new(SharedSegmentSource::new(file_source()))))
        .bench_values(run);
}

#[cfg(not(codspeed))]
#[divan::bench(args = [1, 4, 16], sample_count = 50)]
fn future(bencher: Bencher, threads: usize) {
    bencher
        .with_inputs(|| {
            ready(
                threads,
                Arc::new(SharedSegmentSource::new(FutureSource(file_source()))),
            )
        })
        .bench_values(run);
}
