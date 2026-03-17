// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::session::ArraySession;
use vortex_buffer::ByteBufferMut;
use vortex_fsst::FSSTArray;
use vortex_fsst::canonical::build_views_fast;
use vortex_fsst::decompressor::OptimizedDecompressor;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

/// Generate data that compresses well (few escape codes).
/// Uses a small alphabet that maps entirely to multi-byte FSST symbols.
fn generate_low_escape_data(string_count: usize, avg_len: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let charset = b"abcd";
    let strings: Vec<Option<Box<[u8]>>> = (0..string_count)
        .map(|_| {
            let len = avg_len * rng.random_range(80..=120) / 100;
            let s: Vec<u8> = (0..len)
                .map(|_| charset[rng.random_range(0..charset.len())])
                .collect();
            Some(s.into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable))
}

/// Generate data that compresses poorly (many escape codes).
/// Uses full byte range so most bytes won't be in the symbol table.
fn generate_high_escape_data(string_count: usize, avg_len: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let strings: Vec<Option<Box<[u8]>>> = (0..string_count)
        .map(|_| {
            let len = avg_len * rng.random_range(80..=120) / 100;
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(0..=255u8)).collect();
            Some(s.into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable))
}

/// URL-like data: realistic workload with moderate escape rate.
fn generate_url_like_data(string_count: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let domains = [
        "https://www.example.com",
        "https://api.service.io",
        "http://data.warehouse.net",
        "https://cdn.assets.org",
    ];
    let paths = [
        "/api/v1/users?id=",
        "/search?q=",
        "/catalog/items/",
        "/dashboard/analytics?page=",
    ];
    let strings: Vec<Option<Box<[u8]>>> = (0..string_count)
        .map(|_| {
            let domain = domains[rng.random_range(0..domains.len())];
            let path = paths[rng.random_range(0..paths.len())];
            let id: u32 = rng.random_range(1..100_000);
            let url = format!("{domain}{path}{id}");
            Some(url.into_bytes().into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(strings, DType::Utf8(Nullability::NonNullable))
}

// (string_count, avg_len)
const LOW_ESC_ARGS: &[(usize, usize)] = &[(10_000, 16), (10_000, 64), (10_000, 256), (100_000, 64)];

const HIGH_ESC_ARGS: &[(usize, usize)] =
    &[(10_000, 16), (10_000, 64), (10_000, 256), (100_000, 64)];

const URL_ARGS: &[usize] = &[10_000, 100_000];

static LOW_ESC_ARRAYS: LazyLock<Vec<((usize, usize), FSSTArray)>> = LazyLock::new(|| {
    LOW_ESC_ARGS
        .iter()
        .map(|&(sc, al)| {
            let data = generate_low_escape_data(sc, al);
            let compressor = fsst_train_compressor(&data);
            ((sc, al), fsst_compress(data, &compressor))
        })
        .collect()
});

static HIGH_ESC_ARRAYS: LazyLock<Vec<((usize, usize), FSSTArray)>> = LazyLock::new(|| {
    HIGH_ESC_ARGS
        .iter()
        .map(|&(sc, al)| {
            let data = generate_high_escape_data(sc, al);
            let compressor = fsst_train_compressor(&data);
            ((sc, al), fsst_compress(data, &compressor))
        })
        .collect()
});

static URL_ARRAYS: LazyLock<Vec<(usize, FSSTArray)>> = LazyLock::new(|| {
    URL_ARGS
        .iter()
        .map(|&sc| {
            let data = generate_url_like_data(sc);
            let compressor = fsst_train_compressor(&data);
            (sc, fsst_compress(data, &compressor))
        })
        .collect()
});

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Pre-decompressed data for isolated view-building benchmarks.
struct DecompressedData {
    bytes: Vec<u8>,
    lens: Vec<u64>,
}

fn pre_decompress(encoded: &FSSTArray) -> DecompressedData {
    let compressed = encoded.codes().sliced_bytes();
    let decompressor = OptimizedDecompressor::new(
        encoded.symbols().as_slice(),
        encoded.symbol_lengths().as_slice(),
    );
    let max_cap = encoded
        .decompressor()
        .max_decompression_capacity(compressed.as_slice())
        + 7;
    let mut out = Vec::with_capacity(max_cap);
    let len = decompressor.decompress_into(compressed.as_slice(), out.spare_capacity_mut());
    unsafe { out.set_len(len) };

    let mut ctx = SESSION.create_execution_ctx();
    let uncompressed_lens_array = encoded
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();

    #[allow(clippy::cast_possible_truncation, clippy::unnecessary_cast)]
    let lens: Vec<u64> = match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        uncompressed_lens_array
            .as_slice::<P>()
            .iter()
            .map(|x| *x as u64)
            .collect()
    });

    DecompressedData { bytes: out, lens }
}

static LOW_ESC_DECOMPRESSED: LazyLock<Vec<((usize, usize), DecompressedData)>> =
    LazyLock::new(|| {
        LOW_ESC_ARRAYS
            .iter()
            .map(|(k, arr)| (*k, pre_decompress(arr)))
            .collect()
    });

static HIGH_ESC_DECOMPRESSED: LazyLock<Vec<((usize, usize), DecompressedData)>> =
    LazyLock::new(|| {
        HIGH_ESC_ARRAYS
            .iter()
            .map(|(k, arr)| (*k, pre_decompress(arr)))
            .collect()
    });

static URL_DECOMPRESSED: LazyLock<Vec<(usize, DecompressedData)>> = LazyLock::new(|| {
    URL_ARRAYS
        .iter()
        .map(|(k, arr)| (*k, pre_decompress(arr)))
        .collect()
});

// ============ End-to-end decompress (to_canonical, includes view building) ============

#[divan::bench(args = LOW_ESC_ARGS)]
fn decompress_low_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = LOW_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    bencher
        .with_inputs(|| encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

#[divan::bench(args = HIGH_ESC_ARGS)]
fn decompress_high_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = HIGH_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    bencher
        .with_inputs(|| encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

#[divan::bench(args = URL_ARGS)]
fn decompress_urls(bencher: Bencher, &string_count: &usize) {
    let (_, encoded) = URL_ARRAYS.iter().find(|(k, _)| *k == string_count).unwrap();
    bencher
        .with_inputs(|| encoded)
        .bench_refs(|encoded| encoded.to_canonical());
}

// ============ Isolated view building: old (general build_views) vs new (build_views_fast) ============

#[divan::bench(args = LOW_ESC_ARGS)]
fn views_old_low_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, data) = LOW_ESC_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == args)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &data.lens)
    });
}

#[divan::bench(args = LOW_ESC_ARGS)]
fn views_new_low_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, data) = LOW_ESC_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == args)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views_fast(0, bytes, &data.lens)
    });
}

#[divan::bench(args = HIGH_ESC_ARGS)]
fn views_old_high_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, data) = HIGH_ESC_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == args)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &data.lens)
    });
}

#[divan::bench(args = HIGH_ESC_ARGS)]
fn views_new_high_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, data) = HIGH_ESC_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == args)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views_fast(0, bytes, &data.lens)
    });
}

#[divan::bench(args = URL_ARGS)]
fn views_old_urls(bencher: Bencher, &string_count: &usize) {
    let (_, data) = URL_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == string_count)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &data.lens)
    });
}

#[divan::bench(args = URL_ARGS)]
fn views_new_urls(bencher: Bencher, &string_count: &usize) {
    let (_, data) = URL_DECOMPRESSED
        .iter()
        .find(|(k, _)| *k == string_count)
        .unwrap();
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&data.bytes);
        build_views_fast(0, bytes, &data.lens)
    });
}

// ============ Raw decompress_into: baseline (fsst-rs) vs optimized ============

#[divan::bench(args = LOW_ESC_ARGS)]
fn raw_baseline_low_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = LOW_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    let decompressor = encoded.decompressor();
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = decompressor.max_decompression_capacity(bytes.as_slice()) + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}

#[divan::bench(args = LOW_ESC_ARGS)]
fn raw_optimized_low_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = LOW_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    let decompressor = OptimizedDecompressor::new(
        encoded.symbols().as_slice(),
        encoded.symbol_lengths().as_slice(),
    );
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = encoded
        .decompressor()
        .max_decompression_capacity(bytes.as_slice())
        + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}

#[divan::bench(args = HIGH_ESC_ARGS)]
fn raw_baseline_high_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = HIGH_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    let decompressor = encoded.decompressor();
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = decompressor.max_decompression_capacity(bytes.as_slice()) + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}

#[divan::bench(args = HIGH_ESC_ARGS)]
fn raw_optimized_high_escape(bencher: Bencher, args: (usize, usize)) {
    let (_, encoded) = HIGH_ESC_ARRAYS.iter().find(|(k, _)| *k == args).unwrap();
    let decompressor = OptimizedDecompressor::new(
        encoded.symbols().as_slice(),
        encoded.symbol_lengths().as_slice(),
    );
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = encoded
        .decompressor()
        .max_decompression_capacity(bytes.as_slice())
        + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}

#[divan::bench(args = URL_ARGS)]
fn raw_baseline_urls(bencher: Bencher, &string_count: &usize) {
    let (_, encoded) = URL_ARRAYS.iter().find(|(k, _)| *k == string_count).unwrap();
    let decompressor = encoded.decompressor();
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = decompressor.max_decompression_capacity(bytes.as_slice()) + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}

#[divan::bench(args = URL_ARGS)]
fn raw_optimized_urls(bencher: Bencher, &string_count: &usize) {
    let (_, encoded) = URL_ARRAYS.iter().find(|(k, _)| *k == string_count).unwrap();
    let decompressor = OptimizedDecompressor::new(
        encoded.symbols().as_slice(),
        encoded.symbol_lengths().as_slice(),
    );
    let bytes = encoded.codes().sliced_bytes();
    let max_cap = encoded
        .decompressor()
        .max_decompression_capacity(bytes.as_slice())
        + 7;

    bencher.bench(|| {
        let mut out = Vec::with_capacity(max_cap);
        let len = decompressor.decompress_into(bytes.as_slice(), out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        out
    });
}
