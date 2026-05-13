// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::cast_precision_loss,
    reason = "Benchmark prints ratios as f64."
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::print_stdout,
    clippy::approx_constant,
    clippy::unwrap_used,
    reason = "Example binary: writes to stdout, uses unwrap on infallible operations."
)]

//! Side-by-side compression comparison of HSZ against PCO, BtrBlocks, and
//! Zstd on a handful of synthetic scientific columns. Reports compression
//! ratio, decompression throughput, and the time to answer a range
//! predicate.
//!
//! Run with:
//! ```text
//! cargo run --release -p vortex-hsz --example hsz_vs_pco
//! ```

use std::mem::size_of_val;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use pco::ChunkConfig;
use pco::standalone::simple_compress;
use pco::standalone::simple_decompress;
use vortex::VortexSessionDefault;
use vortex_array::IntoArray as _;
use vortex_array::VortexSessionExecute as _;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::Buffer;
use vortex_hsz::Hsz;
use vortex_hsz::HszConfig;
use vortex_session::VortexSession;

const N: usize = 1_000_000;

/// A smooth signal — represents a temperature or pressure field that varies
/// continuously in space. HSZ's predictor stage should excel here.
fn smooth(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            300.0
                + 25.0 * (t * 6.28318).sin()
                + 5.0 * (t * 18.84956).cos()
                + 0.5 * (t * 125.6637).sin()
        })
        .collect()
}

/// A noisy signal — adds high-frequency white noise on top of the smooth
/// field. Both codecs should still do well; HSZ's residual stage absorbs the
/// noise into quantised integers.
fn noisy(n: usize) -> Vec<f64> {
    let mut state: u64 = 0xdead_beef_dead_beefu64;
    smooth(n)
        .into_iter()
        .map(|v| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((state >> 32) as i32 as f64) / (i32::MAX as f64);
            v + 0.1 * r
        })
        .collect()
}

/// Sparse outliers on top of a smooth field — exercises HSZ's outlier stage.
fn smooth_with_outliers(n: usize) -> Vec<f64> {
    let mut v = smooth(n);
    for i in (0..n).step_by(50_000) {
        v[i] = 1e12;
    }
    v
}

struct Codec<'a> {
    name: &'a str,
    bytes: usize,
    compress_ms: f64,
    decompress_ms: f64,
    sum: f64,
    sum_ms: f64,
    range_count: usize,
    range_ms: f64,
}

fn time<Func: FnOnce() -> Out, Out>(func: Func) -> (Out, Duration) {
    let start = Instant::now();
    let result = func();
    (result, start.elapsed())
}

fn run_hsz(name: &'static str, data: &[f64], eps: f64) -> Codec<'static> {
    let cfg = HszConfig { eps };
    let (hsz, t_compress) = time(|| Hsz::compress(data, cfg).unwrap());
    let bytes = hsz.encoded_bytes();
    let (_, t_decompress) = time(|| hsz.decompress());
    let (sum, t_sum) = time(|| hsz.sum());
    let (range_count, t_range) = time(|| {
        let (mask, _) = hsz.between_mask(290.0, 310.0);
        mask.true_count()
    });
    Codec {
        name,
        bytes,
        compress_ms: t_compress.as_secs_f64() * 1e3,
        decompress_ms: t_decompress.as_secs_f64() * 1e3,
        sum,
        sum_ms: t_sum.as_secs_f64() * 1e3,
        range_count,
        range_ms: t_range.as_secs_f64() * 1e3,
    }
}

fn run_pco(name: &'static str, data: &[f64], level: usize) -> Codec<'static> {
    let cfg = ChunkConfig::default().with_compression_level(level);
    let (compressed, t_compress) = time(|| simple_compress(data, &cfg).unwrap());
    let bytes = compressed.len();
    let (decoded, t_decompress) = time(|| simple_decompress::<f64>(&compressed).unwrap());
    let (sum, t_sum) = time(|| decoded.iter().copied().sum::<f64>());
    let (range_count, t_range) = time(|| {
        decoded
            .iter()
            .copied()
            .filter(|v| *v >= 290.0 && *v <= 310.0)
            .count()
    });
    Codec {
        name,
        bytes,
        compress_ms: t_compress.as_secs_f64() * 1e3,
        decompress_ms: t_decompress.as_secs_f64() * 1e3,
        sum,
        sum_ms: t_sum.as_secs_f64() * 1e3,
        range_count,
        range_ms: t_range.as_secs_f64() * 1e3,
    }
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(<VortexSession as VortexSessionDefault>::default);

fn run_btrblocks(
    name: &'static str,
    compressor: &BtrBlocksCompressor,
    data: &[f64],
) -> Codec<'static> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(Buffer::copy_from(data), Validity::NonNullable).into_array();
    let (compressed, t_compress) = time(|| compressor.compress(&array, &mut ctx).unwrap());
    let bytes = compressed.nbytes() as usize;
    let (decoded, t_decompress) = time(|| {
        compressed
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
    });
    let decoded_slice = decoded.to_buffer::<f64>();
    let (sum, t_sum) = time(|| decoded_slice.as_slice().iter().copied().sum::<f64>());
    let (range_count, t_range) = time(|| {
        decoded_slice
            .as_slice()
            .iter()
            .copied()
            .filter(|v| *v >= 290.0 && *v <= 310.0)
            .count()
    });
    Codec {
        name,
        bytes,
        compress_ms: t_compress.as_secs_f64() * 1e3,
        decompress_ms: t_decompress.as_secs_f64() * 1e3,
        sum,
        sum_ms: t_sum.as_secs_f64() * 1e3,
        range_count,
        range_ms: t_range.as_secs_f64() * 1e3,
    }
}

fn run_zstd(name: &'static str, data: &[f64], level: i32) -> Codec<'static> {
    let raw = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
    let (compressed, t_compress) = time(|| zstd::stream::encode_all(raw, level).unwrap());
    let bytes = compressed.len();
    let (decoded_bytes, t_decompress) =
        time(|| zstd::stream::decode_all(compressed.as_slice()).unwrap());
    let decoded: &[f64] = unsafe {
        std::slice::from_raw_parts(
            decoded_bytes.as_ptr().cast::<f64>(),
            decoded_bytes.len() / size_of::<f64>(),
        )
    };
    let (sum, t_sum) = time(|| decoded.iter().copied().sum::<f64>());
    let (range_count, t_range) = time(|| {
        decoded
            .iter()
            .copied()
            .filter(|v| *v >= 290.0 && *v <= 310.0)
            .count()
    });
    Codec {
        name,
        bytes,
        compress_ms: t_compress.as_secs_f64() * 1e3,
        decompress_ms: t_decompress.as_secs_f64() * 1e3,
        sum,
        sum_ms: t_sum.as_secs_f64() * 1e3,
        range_count,
        range_ms: t_range.as_secs_f64() * 1e3,
    }
}

fn report(dataset: &str, raw_bytes: usize, codecs: &[Codec<'_>]) {
    println!();
    println!("=== {dataset} ({} rows, {} raw bytes) ===", N, raw_bytes);
    println!(
        "{:<10} {:>12} {:>8} {:>11} {:>13} {:>16} {:>15} {:>9}",
        "codec",
        "bytes",
        "ratio",
        "compress",
        "decompress",
        "sum (ms / val)",
        "range (ms)",
        "range#"
    );
    for c in codecs {
        let ratio = raw_bytes as f64 / c.bytes as f64;
        println!(
            "{:<10} {:>12} {:>7.2}x {:>9.2}ms {:>11.2}ms {:>9.2}ms / {:>10.3e} {:>13.2}ms {:>9}",
            c.name,
            c.bytes,
            ratio,
            c.compress_ms,
            c.decompress_ms,
            c.sum_ms,
            c.sum,
            c.range_ms,
            c.range_count,
        );
    }
}

fn main() {
    println!("HSZ vs PCO vs BtrBlocks vs Zstd ({} rows of f64)", N);
    let raw_bytes = N * size_of::<f64>();
    let btr_default = BtrBlocksCompressor::default();
    // BtrBlocks "compact" enables PCO as a candidate scheme inside the
    // cascading compressor.
    let btr_compact = BtrBlocksCompressorBuilder::default().with_compact().build();

    for (name, data, eps) in [
        ("smooth", smooth(N), 1e-3),
        ("noisy", noisy(N), 5e-2),
        ("outliers", smooth_with_outliers(N), 1e-3),
    ] {
        let hsz = run_hsz("hsz", &data, eps);
        let pco_lo = run_pco("pco/l4", &data, 4);
        let pco_hi = run_pco("pco/l8", &data, 8);
        let btr = run_btrblocks("btr", &btr_default, &data);
        let btr_c = run_btrblocks("btr/compact", &btr_compact, &data);
        let zstd_lo = run_zstd("zstd/l3", &data, 3);
        let zstd_hi = run_zstd("zstd/l9", &data, 9);
        report(
            name,
            raw_bytes,
            &[hsz, pco_lo, pco_hi, btr, btr_c, zstd_lo, zstd_hi],
        );
    }

    println!();
    println!(
        "Notes: HSZ is lossy within `eps`; PCO, BtrBlocks, and Zstd are \
         lossless. HSZ answers `sum` from Stage-0 block summaries without \
         touching residuals; the other codecs run sum over the canonical \
         buffer. `range` is `count(290 <= x <= 310)`; HSZ uses zone-map \
         skipping, the others scan the decoded buffer."
    );
}
