// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::expect_used)]

mod bench_config;
// Unused here but suppresses dead_code warning for the shared module.
const _: &[(usize, &str)] = bench_config::BENCH_SIZES;

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use cudarc::driver::HostSlice;
use cudarc::driver::SyncOnDrop;
use cudarc::driver::result;
use cudarc::driver::sys;
use cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use cudarc::driver::sys::CUmemPool_attribute_enum;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const BUFFER_SIZE: usize = 1024 * 1024 * 1024;
const BUFFER_SIZE_NAME: &str = "1GiB";
const DEVICE_MEM_POOL_RELEASE_THRESHOLD_PERCENT: usize = 50;
const CPU_WORK_DURATION: Duration = Duration::from_millis(4);

const HOST_MEMORY_KINDS: &[(&str, Option<u32>)] = &[
    // Pageable host memory allocated through the Rust global allocator. CUDA may need to stage or
    // pin pages internally before the host-to-device copy can run.
    ("pageable", None),
    // Page-locked host memory from cuMemHostAlloc with no additional flags.
    ("pinned_default", Some(0)),
    // Page-locked write-combined host memory. This favors CPU writes into the source buffer but
    // makes CPU reads from it expensive.
    ("pinned_write_combined", Some(CU_MEMHOSTALLOC_WRITECOMBINED)),
];

fn synthetic_cpu_work(duration: Duration) {
    let start = Instant::now();
    let mut work = 0_u64;

    while start.elapsed() < duration {
        work = std::hint::black_box(work.wrapping_mul(31).wrapping_add(1));
    }

    std::hint::black_box(work);
}

struct CudaHostBuffer {
    ctx: Arc<CudaContext>,
    ptr: *mut u8,
    len: usize,
}

// TODO(0ax1): Move CudaHostBuffer out of the test logic and make
// explicit allocation with flags part of the vortex-cuda API.
impl CudaHostBuffer {
    fn alloc(ctx: &Arc<CudaContext>, len: usize, flags: u32) -> Self {
        ctx.bind_to_thread().expect("bind cuda context");
        let ptr = unsafe { result::malloc_host(len, flags) }.expect("allocate cuda host buffer");
        Self {
            ctx: Arc::clone(ctx),
            ptr: ptr.cast(),
            len,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl HostSlice<u8> for CudaHostBuffer {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }
}

impl Drop for CudaHostBuffer {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { result::free_host(self.ptr.cast()) });
    }
}

fn device_mem_pool_release_threshold(pool: sys::CUmemoryPool) -> u64 {
    let mut threshold = 0_u64;
    unsafe {
        result::mem_pool::get_attribute(
            pool,
            CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&raw mut threshold).cast(),
        )
    }
    .expect("get cuda memory pool release threshold");
    threshold
}

fn set_device_mem_pool_release_threshold(pool: sys::CUmemoryPool, threshold: u64) {
    let mut threshold = threshold;
    unsafe {
        result::mem_pool::set_attribute(
            pool,
            CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&raw mut threshold).cast(),
        )
    }
    .expect("set cuda memory pool release threshold");
}

fn cuda_default_mem_pool() -> sys::CUmemoryPool {
    let device = result::device::get(0).expect("cuda device");
    unsafe { result::device::get_mem_pool(device) }.expect("cuda device memory pool")
}

// The CUDA default mempool setting is shared across this benchmark process.
// Restore it so later benchmark groups do not inherit the tuned threshold.
struct ThresholdGuard {
    pool: sys::CUmemoryPool,
    original_threshold: u64,
}

impl ThresholdGuard {
    fn set(pool: sys::CUmemoryPool, threshold: u64) -> Self {
        let original_threshold = device_mem_pool_release_threshold(pool);
        set_device_mem_pool_release_threshold(pool, threshold);
        Self {
            pool,
            original_threshold,
        }
    }
}

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        set_device_mem_pool_release_threshold(self.pool, self.original_threshold);
    }
}

fn high_device_mem_pool_release_threshold(ctx: &CudaContext) -> u64 {
    let (_, total_memory) = ctx.mem_get_info().expect("cuda memory info");

    // Cap retention at 50% of device memory so a peak allocation does not let the
    // pool unnecessarily hold the entire GPU and increase OOM/coexistence risk.
    (total_memory / 100 * DEVICE_MEM_POOL_RELEASE_THRESHOLD_PERCENT) as u64
}

fn benchmark_core_primitives(c: &mut Criterion) {
    // Measures steady-state host-call latency for CUDA device allocation strategies after
    // each allocation has been returned to the pool and the stream has synchronized.
    let mut device_alloc_group = c.benchmark_group("cuda/core_primitives/device_alloc_reuse");

    device_alloc_group.bench_with_input(
        BenchmarkId::new("default_pool", BUFFER_SIZE_NAME),
        &BUFFER_SIZE,
        |b, &size| {
            let cuda_ctx = CudaContext::new(0).expect("cuda ctx");
            let stream = cuda_ctx.new_stream().expect("cuda stream");

            // Seed the pool so the timed loop measures reuse after free+sync.
            let dest = unsafe { stream.alloc::<u8>(size) }.expect("allocate device buffer");
            drop(dest);
            stream.synchronize().expect("synchronize stream");

            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;

                for _ in 0..iters {
                    let start = Instant::now();
                    let dest = unsafe { stream.alloc::<u8>(size) }.expect("allocate device buffer");
                    elapsed += start.elapsed();

                    drop(dest);
                    stream.synchronize().expect("synchronize stream");
                }

                elapsed
            });
        },
    );

    device_alloc_group.bench_with_input(
        BenchmarkId::new("default_pool_75pct_threshold", BUFFER_SIZE_NAME),
        &BUFFER_SIZE,
        |b, &size| {
            let cuda_ctx = CudaContext::new(0).expect("cuda ctx");
            let stream = cuda_ctx.new_stream().expect("cuda stream");

            let pool = cuda_default_mem_pool();
            let _release_threshold_guard =
                ThresholdGuard::set(pool, high_device_mem_pool_release_threshold(&cuda_ctx));

            // Seed the pool so the timed loop measures reuse after free+sync.
            let dest = unsafe { stream.alloc::<u8>(size) }.expect("allocate device buffer");
            drop(dest);
            stream.synchronize().expect("synchronize stream");

            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;

                for _ in 0..iters {
                    let start = Instant::now();
                    let dest = unsafe { stream.alloc::<u8>(size) }.expect("allocate device buffer");
                    elapsed += start.elapsed();

                    drop(dest);
                    stream.synchronize().expect("synchronize stream");
                }

                elapsed
            });
        },
    );

    device_alloc_group.finish();

    // Measures a synchronized host-to-device copy after both host source and device
    // destination have already been allocated and the source has been initialized.
    // This isolates copy throughput for each host allocation mode as much as possible.
    let mut copy_group =
        c.benchmark_group("cuda/core_primitives/allocated_host_to_device_copy_and_sync");

    copy_group.throughput(Throughput::Bytes(BUFFER_SIZE as u64));

    for &(name, flags) in HOST_MEMORY_KINDS {
        copy_group.bench_with_input(
            BenchmarkId::new(name, BUFFER_SIZE_NAME),
            &BUFFER_SIZE,
            |b, &size| {
                let cuda_ctx = CudaContext::new(0).expect("cuda ctx");
                let stream = cuda_ctx.new_stream().expect("cuda stream");

                match flags {
                    Some(flags) => b.iter_batched(
                        || {
                            let mut source = CudaHostBuffer::alloc(&cuda_ctx, size, flags);
                            source.as_mut_slice().fill(0xA5);
                            let dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            (source, dest)
                        },
                        |(source, mut dest)| {
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                    None => b.iter_batched(
                        || {
                            let mut source = vec![0u8; size];
                            source.fill(0xA5);
                            let dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            (source, dest)
                        },
                        |(source, mut dest)| {
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                }
            },
        );
    }

    copy_group.finish();

    // Measures host-to-device copy with a fixed CPU-side workload between enqueue
    // and synchronization. Pinned host memory should let memcpy_htod return after
    // enqueueing the DMA, so more of the CPU work can overlap the transfer.
    let mut overlap_group = c.benchmark_group(
        "cuda/core_primitives/allocated_host_to_device_copy_4ms_cpu_work_then_sync",
    );

    overlap_group.throughput(Throughput::Bytes(BUFFER_SIZE as u64));

    for &(name, flags) in HOST_MEMORY_KINDS {
        overlap_group.bench_with_input(
            BenchmarkId::new(name, BUFFER_SIZE_NAME),
            &BUFFER_SIZE,
            |b, &size| {
                let cuda_ctx = CudaContext::new(0).expect("cuda ctx");
                let stream = cuda_ctx.new_stream().expect("cuda stream");

                match flags {
                    Some(flags) => b.iter_batched(
                        || {
                            let mut source = CudaHostBuffer::alloc(&cuda_ctx, size, flags);
                            source.as_mut_slice().fill(0xA5);
                            let dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            (source, dest)
                        },
                        |(source, mut dest)| {
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            synthetic_cpu_work(CPU_WORK_DURATION);
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                    None => b.iter_batched(
                        || {
                            let mut source = vec![0u8; size];
                            source.fill(0xA5);
                            let dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            (source, dest)
                        },
                        |(source, mut dest)| {
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            synthetic_cpu_work(CPU_WORK_DURATION);
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                }
            },
        );
    }

    overlap_group.finish();

    // Measures device allocation plus host-to-device copy. Host source allocation and
    // initialization stay in Criterion setup, so this separates device allocation cost
    // from host allocation cost.
    let mut alloc_copy_group =
        c.benchmark_group("cuda/core_primitives/device_alloc_host_to_device_copy_and_sync");

    alloc_copy_group.throughput(Throughput::Bytes(BUFFER_SIZE as u64));

    for &(name, flags) in HOST_MEMORY_KINDS {
        alloc_copy_group.bench_with_input(
            BenchmarkId::new(name, BUFFER_SIZE_NAME),
            &BUFFER_SIZE,
            |b, &size| {
                let cuda_ctx = CudaContext::new(0).expect("cuda ctx");
                let stream = cuda_ctx.new_stream().expect("cuda stream");

                match flags {
                    Some(flags) => b.iter_batched(
                        || {
                            let mut source = CudaHostBuffer::alloc(&cuda_ctx, size, flags);
                            source.as_mut_slice().fill(0xA5);
                            source
                        },
                        |source| {
                            let mut dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                    None => b.iter_batched(
                        || {
                            let mut source = vec![0u8; size];
                            source.fill(0xA5);
                            source
                        },
                        |source| {
                            let mut dest = unsafe { stream.alloc::<u8>(size) }
                                .expect("allocate device buffer");
                            stream.memcpy_htod(&source, &mut dest).expect("memcpy_htod");
                            stream.synchronize().expect("synchronize stream");
                        },
                        BatchSize::PerIteration,
                    ),
                }
            },
        );
    }

    alloc_copy_group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_core_primitives
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
