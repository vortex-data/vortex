// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for OnPair decompression.
//!
//! Compares CPU `Column::decode_all()` against the GPU `onpair_gpu_decode_all`
//! executor over ClickBench-style URL data. Both timings are wall-clock and
//! include the same compressed input → flat `(bytes, offsets)` output path:
//!
//! * CPU: `Column::decode_all()` — runs the two-pass CPU decoder
//!   (length-sum + decode) from `onpair-lib`.
//! * GPU: host-side length-sum + offset prefix-sum + H2D copies + kernel
//!   launch + D2H of decoded bytes. The host length-sum is the same Big-O
//!   work the CPU pass-1 does, so the two sides do comparable work.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

#[allow(dead_code)]
mod bench_config;

use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use onpair_lib::Column;
use onpair_lib::DEFAULT_DICT12_CONFIG;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::onpair_gpu_decode_all;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::test_utils::generate_clickbench_urls;

// Per-input sizes. URLs average ~80–120 bytes; 1M rows ≈ 100 MB of raw text.
// Stay below the 10M FSST bench size — OnPair training is heavier (BPE merge
// loop) and the bench's first thing we measure is the train+compress.
const BENCH_SIZES: &[(usize, &str)] = &[(1_000_000, "1M")];

/// Pack `strings` into the flat `(bytes, offsets[len + 1])` shape
/// `Column::compress` expects.
fn pack_strings(strings: &[String]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::new();
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u64);
    for s in strings {
        bytes.extend_from_slice(s.as_bytes());
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn benchmark_onpair_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        // Train + compress once per input size, outside the timing loop.
        let strings = generate_clickbench_urls(n);
        let (bytes, offsets) = pack_strings(&strings);
        let uncompressed_size = bytes.len() as u64;
        let column =
            Column::compress(&bytes, &offsets, DEFAULT_DICT12_CONFIG).expect("OnPair compress");

        group.throughput(Throughput::Bytes(uncompressed_size));

        // ── CPU baseline: Column::decode_all ───────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("onpair/cpu_decode", len_str),
            &column,
            |b, column: &Column| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let (decoded_bytes, decoded_offsets) = column.decode_all();
                        criterion::black_box((decoded_bytes, decoded_offsets));
                    }
                    start.elapsed()
                });
            },
        );

        // ── GPU: onpair_gpu_decode_all (incl. H2D + kernel + D2H) ──────────
        group.bench_with_input(
            BenchmarkId::new("onpair/gpu_decode", len_str),
            &column,
            |b, column: &Column| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .expect("failed to create execution context");
                    let parts = column.parts().expect("Column::parts");
                    let start = Instant::now();
                    for _ in 0..iters {
                        let decoded =
                            block_on(onpair_gpu_decode_all(&parts, &mut cuda_ctx)).unwrap();
                        criterion::black_box(decoded);
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config().measurement_time(Duration::from_secs(5));
    targets = benchmark_onpair_decode
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
