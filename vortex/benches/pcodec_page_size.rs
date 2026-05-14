// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare pcodec with the default page size against pcodec with a 1024-value
//! ("1k") page size on `f64` and `i64` data.
//!
//! Pcodec lets you tune the page size independently of the chunk size. Smaller
//! pages add per-page overhead (so the compression ratio gets worse) but let
//! random access decode a much smaller window, which should help `scalar_at`.
//! These benches measure both effects.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_precision_loss)]

use std::sync::LazyLock;

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::session::ArraySession;
use vortex::encodings::pco::Pco;
use vortex::encodings::pco::PcoArray;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const NUM_VALUES: usize = 100_000;
const NUM_ACCESSES: usize = 1000;

/// `0` tells `Pco::from_primitive` to fall back to the chunk-default page size
/// (`pco::DEFAULT_MAX_PAGE_N`, currently 262144 values).
const DEFAULT_PAGE: usize = 0;
const PAGE_1K: usize = 1024;
const COMPRESSION_LEVEL: usize = 3;

fn setup_arrays() -> (PrimitiveArray, PrimitiveArray) {
    let mut rng = StdRng::seed_from_u64(0);
    // A noisy ramp: large dynamic range with smooth structure - the kind of
    // data where pcodec's bin/offset scheme actually does work, so the page
    // overhead is visible in the compression ratio.
    let f64_array = PrimitiveArray::from_iter((0..NUM_VALUES).map(|i| {
        let noise: f64 = rng.random_range(-0.5..0.5);
        i as f64 * 0.001 + noise
    }));
    let i64_array =
        PrimitiveArray::from_iter((0..NUM_VALUES).map(|i| i as i64 + rng.random_range(-50..50)));
    (f64_array, i64_array)
}

fn compress(parray: &PrimitiveArray, values_per_page: usize) -> PcoArray {
    let mut ctx = SESSION.create_execution_ctx();
    Pco::from_primitive(
        parray.as_view(),
        COMPRESSION_LEVEL,
        values_per_page,
        &mut ctx,
    )
    .unwrap()
}

fn print_compression_ratios() {
    let (f64_array, i64_array) = setup_arrays();
    let f64_uncompressed = f64_array.nbytes();
    let i64_uncompressed = i64_array.nbytes();

    let f64_default = compress(&f64_array, DEFAULT_PAGE).nbytes();
    let f64_1k = compress(&f64_array, PAGE_1K).nbytes();
    let i64_default = compress(&i64_array, DEFAULT_PAGE).nbytes();
    let i64_1k = compress(&i64_array, PAGE_1K).nbytes();

    eprintln!();
    eprintln!(
        "pcodec page-size compression ratio ({NUM_VALUES} values, level {COMPRESSION_LEVEL}):"
    );
    eprintln!("  f64  uncompressed = {:>10} bytes", f64_uncompressed);
    eprintln!(
        "  f64  default page = {:>10} bytes  ({:.3}x, {:.2} bits/value)",
        f64_default,
        f64_uncompressed as f64 / f64_default as f64,
        (f64_default as f64 * 8.0) / NUM_VALUES as f64,
    );
    eprintln!(
        "  f64  1k      page = {:>10} bytes  ({:.3}x, {:.2} bits/value)",
        f64_1k,
        f64_uncompressed as f64 / f64_1k as f64,
        (f64_1k as f64 * 8.0) / NUM_VALUES as f64,
    );
    eprintln!("  i64  uncompressed = {:>10} bytes", i64_uncompressed);
    eprintln!(
        "  i64  default page = {:>10} bytes  ({:.3}x, {:.2} bits/value)",
        i64_default,
        i64_uncompressed as f64 / i64_default as f64,
        (i64_default as f64 * 8.0) / NUM_VALUES as f64,
    );
    eprintln!(
        "  i64  1k      page = {:>10} bytes  ({:.3}x, {:.2} bits/value)",
        i64_1k,
        i64_uncompressed as f64 / i64_1k as f64,
        (i64_1k as f64 * 8.0) / NUM_VALUES as f64,
    );
    eprintln!();
}

fn with_byte_counter<'a, 'b>(bencher: Bencher<'a, 'b>, bytes: u64) -> Bencher<'a, 'b> {
    #[cfg(not(codspeed))]
    return bencher.counter(BytesCount::new(bytes));
    #[cfg(codspeed)]
    {
        _ = bytes;
        return bencher;
    }
}

fn random_indices() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..NUM_VALUES))
        .collect()
}

fn main() {
    print_compression_ratios();
    divan::main();
}

// --- compression ---

#[divan::bench(name = "pcodec_compress_f64_default_page")]
fn bench_compress_f64_default(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&f64_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            Pco::from_primitive(a.as_view(), COMPRESSION_LEVEL, DEFAULT_PAGE, ctx).unwrap()
        });
}

#[divan::bench(name = "pcodec_compress_f64_1k_page")]
fn bench_compress_f64_1k(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&f64_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            Pco::from_primitive(a.as_view(), COMPRESSION_LEVEL, PAGE_1K, ctx).unwrap()
        });
}

#[divan::bench(name = "pcodec_compress_i64_default_page")]
fn bench_compress_i64_default(bencher: Bencher) {
    let (_, i64_array) = setup_arrays();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&i64_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            Pco::from_primitive(a.as_view(), COMPRESSION_LEVEL, DEFAULT_PAGE, ctx).unwrap()
        });
}

#[divan::bench(name = "pcodec_compress_i64_1k_page")]
fn bench_compress_i64_1k(bencher: Bencher) {
    let (_, i64_array) = setup_arrays();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&i64_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            Pco::from_primitive(a.as_view(), COMPRESSION_LEVEL, PAGE_1K, ctx).unwrap()
        });
}

// --- decompression (full) ---

#[divan::bench(name = "pcodec_decompress_f64_default_page")]
fn bench_decompress_f64_default(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    let compressed = compress(&f64_array, DEFAULT_PAGE).into_array();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| (**a).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

#[divan::bench(name = "pcodec_decompress_f64_1k_page")]
fn bench_decompress_f64_1k(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    let compressed = compress(&f64_array, PAGE_1K).into_array();
    with_byte_counter(bencher, (NUM_VALUES * 8) as u64)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| (**a).clone().execute::<PrimitiveArray>(ctx).unwrap());
}

// --- scalar_at ---

#[divan::bench(name = "pcodec_scalar_at_f64_default_page")]
fn bench_scalar_at_f64_default(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    let compressed = compress(&f64_array, DEFAULT_PAGE).into_array();
    let indices = random_indices();
    bencher
        .with_inputs(|| (&compressed, &indices))
        .bench_refs(|(array, indices)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for &idx in indices.iter() {
                divan::black_box(array.execute_scalar(idx, &mut ctx).unwrap());
            }
        });
}

#[divan::bench(name = "pcodec_scalar_at_f64_1k_page")]
fn bench_scalar_at_f64_1k(bencher: Bencher) {
    let (f64_array, _) = setup_arrays();
    let compressed = compress(&f64_array, PAGE_1K).into_array();
    let indices = random_indices();
    bencher
        .with_inputs(|| (&compressed, &indices))
        .bench_refs(|(array, indices)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for &idx in indices.iter() {
                divan::black_box(array.execute_scalar(idx, &mut ctx).unwrap());
            }
        });
}

#[divan::bench(name = "pcodec_scalar_at_i64_default_page")]
fn bench_scalar_at_i64_default(bencher: Bencher) {
    let (_, i64_array) = setup_arrays();
    let compressed = compress(&i64_array, DEFAULT_PAGE).into_array();
    let indices = random_indices();
    bencher
        .with_inputs(|| (&compressed, &indices))
        .bench_refs(|(array, indices)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for &idx in indices.iter() {
                divan::black_box(array.execute_scalar(idx, &mut ctx).unwrap());
            }
        });
}

#[divan::bench(name = "pcodec_scalar_at_i64_1k_page")]
fn bench_scalar_at_i64_1k(bencher: Bencher) {
    let (_, i64_array) = setup_arrays();
    let compressed = compress(&i64_array, PAGE_1K).into_array();
    let indices = random_indices();
    bencher
        .with_inputs(|| (&compressed, &indices))
        .bench_refs(|(array, indices)| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for &idx in indices.iter() {
                divan::black_box(array.execute_scalar(idx, &mut ctx).unwrap());
            }
        });
}
