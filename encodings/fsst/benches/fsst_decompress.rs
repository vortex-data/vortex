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
use vortex_fsst::canonical::VIEW_BUILD_PADDING;
use vortex_fsst::canonical::build_views_fast;
use vortex_fsst::decompressor::OptimizedDecompressor;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::test_utils;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// Session for executing lazy expressions
// ---------------------------------------------------------------------------

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

/// Short strings (3-12 bytes), all ≤ BinaryView::MAX_INLINED_SIZE.
/// Exercises the inlined-view path exclusively.
fn generate_short_strings(count: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let words: &[&[u8]] = &[
        b"the", b"and", b"for", b"are", b"but", b"not", b"you", b"all", b"can", b"had", b"her",
        b"was", b"one", b"our", b"out", b"day", b"get", b"has", b"him", b"his", b"how", b"its",
        b"may", b"new", b"now", b"old", b"see", b"way", b"who", b"did", b"oil", b"sit", b"cat",
        b"dog", b"red", b"big", b"top", b"sun", b"run", b"hot", b"yes", b"far", b"ask", b"own",
        b"say", b"low", b"key", b"few",
    ];
    let strings: Vec<Option<Box<[u8]>>> = (0..count)
        .map(|_| {
            // 1-3 words concatenated, always ≤ 12 bytes
            let nwords = rng.random_range(1..=3usize);
            let mut buf = Vec::with_capacity(12);
            for idx in 0..nwords {
                if idx > 0 {
                    buf.push(b'-');
                }
                let word = words[rng.random_range(0..words.len())];
                if buf.len() + word.len() + usize::from(idx > 0) > 12 {
                    break;
                }
                buf.extend_from_slice(word);
            }
            Some(buf.into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable))
}

/// Medium strings (8-20 bytes), mix of inlined and reference views.
/// Straddles the 12-byte BinaryView inlining threshold.
fn generate_medium_strings(count: usize) -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);
    let prefixes: &[&[u8]] = &[
        b"usr_", b"grp_", b"tok_", b"ses_", b"evt_", b"req_", b"txn_", b"msg_",
    ];
    let strings: Vec<Option<Box<[u8]>>> = (0..count)
        .map(|_| {
            let prefix = prefixes[rng.random_range(0..prefixes.len())];
            let suffix_len = rng.random_range(4..=16usize);
            let mut buf = Vec::with_capacity(prefix.len() + suffix_len);
            buf.extend_from_slice(prefix);
            for _ in 0..suffix_len {
                buf.push(rng.random_range(b'a'..=b'z'));
            }
            Some(buf.into_boxed_slice())
        })
        .collect();
    VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable))
}

fn make_fsst(data: VarBinArray) -> FSSTArray {
    let compressor = fsst_train_compressor(&data);
    fsst_compress(data, &compressor)
}

// ---------------------------------------------------------------------------
// Lazy-initialized datasets: real-world from test_utils + custom short/medium
// ---------------------------------------------------------------------------

const N: usize = 100_000;

static SHORT_STRINGS: LazyLock<FSSTArray> = LazyLock::new(|| make_fsst(generate_short_strings(N)));
static MEDIUM_STRINGS: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst(generate_medium_strings(N)));
static EMAILS: LazyLock<FSSTArray> = LazyLock::new(|| test_utils::make_fsst_emails(N));
static SHORT_URLS: LazyLock<FSSTArray> = LazyLock::new(|| test_utils::make_fsst_short_urls(N));
static CLICKBENCH_URLS: LazyLock<FSSTArray> =
    LazyLock::new(|| test_utils::make_fsst_clickbench_urls(N));
static LOG_LINES: LazyLock<FSSTArray> = LazyLock::new(|| test_utils::make_fsst_log_lines(N));
static JSON_STRINGS: LazyLock<FSSTArray> = LazyLock::new(|| test_utils::make_fsst_json_strings(N));
static FILE_PATHS: LazyLock<FSSTArray> = LazyLock::new(|| test_utils::make_fsst_file_paths(N));

// ---------------------------------------------------------------------------
// Pre-decompressed data for isolated view-building benchmarks
// ---------------------------------------------------------------------------

struct DecompressedData {
    bytes: Vec<u8>,
    lens: Vec<u64>,
}

/// Create a padded `ByteBufferMut` from a byte slice, with extra capacity for safe
/// 16-byte unaligned reads in `build_views_fast`.
fn padded_buffer(data: &[u8]) -> ByteBufferMut {
    let mut buf = ByteBufferMut::with_capacity(data.len() + VIEW_BUILD_PADDING);
    buf.extend_from_slice(data);
    buf
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

static SHORT_STRINGS_DEC: LazyLock<DecompressedData> =
    LazyLock::new(|| pre_decompress(&SHORT_STRINGS));
static MEDIUM_STRINGS_DEC: LazyLock<DecompressedData> =
    LazyLock::new(|| pre_decompress(&MEDIUM_STRINGS));
static EMAILS_DEC: LazyLock<DecompressedData> = LazyLock::new(|| pre_decompress(&EMAILS));
static SHORT_URLS_DEC: LazyLock<DecompressedData> = LazyLock::new(|| pre_decompress(&SHORT_URLS));
static CLICKBENCH_URLS_DEC: LazyLock<DecompressedData> =
    LazyLock::new(|| pre_decompress(&CLICKBENCH_URLS));
static LOG_LINES_DEC: LazyLock<DecompressedData> = LazyLock::new(|| pre_decompress(&LOG_LINES));
static JSON_STRINGS_DEC: LazyLock<DecompressedData> =
    LazyLock::new(|| pre_decompress(&JSON_STRINGS));
static FILE_PATHS_DEC: LazyLock<DecompressedData> = LazyLock::new(|| pre_decompress(&FILE_PATHS));

// ============================================================================
// End-to-end decompress (to_canonical): measures full pipeline
// ============================================================================

#[divan::bench]
fn e2e_short_strings(bencher: Bencher) {
    let arr = &*SHORT_STRINGS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_medium_strings(bencher: Bencher) {
    let arr = &*MEDIUM_STRINGS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_emails(bencher: Bencher) {
    let arr = &*EMAILS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_short_urls(bencher: Bencher) {
    let arr = &*SHORT_URLS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_clickbench_urls(bencher: Bencher) {
    let arr = &*CLICKBENCH_URLS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_log_lines(bencher: Bencher) {
    let arr = &*LOG_LINES;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_json_strings(bencher: Bencher) {
    let arr = &*JSON_STRINGS;
    bencher.bench(|| arr.to_canonical());
}

#[divan::bench]
fn e2e_file_paths(bencher: Bencher) {
    let arr = &*FILE_PATHS;
    bencher.bench(|| arr.to_canonical());
}

// ============================================================================
// Isolated view building: old (general build_views) vs new (build_views_fast)
// ============================================================================

// --- Short strings (≤12 bytes, all inlined) ---

#[divan::bench]
fn views_old_short_strings(bencher: Bencher) {
    let d = &*SHORT_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_short_strings(bencher: Bencher) {
    let d = &*SHORT_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- Medium strings (8-20 bytes, mix of inlined and reference) ---

#[divan::bench]
fn views_old_medium_strings(bencher: Bencher) {
    let d = &*MEDIUM_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_medium_strings(bencher: Bencher) {
    let d = &*MEDIUM_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- Emails (~20 bytes, all reference) ---

#[divan::bench]
fn views_old_emails(bencher: Bencher) {
    let d = &*EMAILS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_emails(bencher: Bencher) {
    let d = &*EMAILS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- Short URLs (~35 bytes) ---

#[divan::bench]
fn views_old_short_urls(bencher: Bencher) {
    let d = &*SHORT_URLS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_short_urls(bencher: Bencher) {
    let d = &*SHORT_URLS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- ClickBench URLs (~80-120 bytes) ---

#[divan::bench]
fn views_old_clickbench_urls(bencher: Bencher) {
    let d = &*CLICKBENCH_URLS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_clickbench_urls(bencher: Bencher) {
    let d = &*CLICKBENCH_URLS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- Log lines (~120+ bytes) ---

#[divan::bench]
fn views_old_log_lines(bencher: Bencher) {
    let d = &*LOG_LINES_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_log_lines(bencher: Bencher) {
    let d = &*LOG_LINES_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- JSON strings (~80+ bytes) ---

#[divan::bench]
fn views_old_json_strings(bencher: Bencher) {
    let d = &*JSON_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_json_strings(bencher: Bencher) {
    let d = &*JSON_STRINGS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// --- File paths (~30-60 bytes) ---

#[divan::bench]
fn views_old_file_paths(bencher: Bencher) {
    let d = &*FILE_PATHS_DEC;
    bencher.bench(|| {
        let bytes = ByteBufferMut::copy_from(&d.bytes);
        build_views(0, MAX_BUFFER_LEN, bytes, &d.lens)
    });
}

#[divan::bench]
fn views_new_file_paths(bencher: Bencher) {
    let d = &*FILE_PATHS_DEC;
    bencher.bench(|| {
        let bytes = padded_buffer(&d.bytes);
        build_views_fast(0, bytes, &d.lens)
    });
}

// ============================================================================
// Raw decompress_into: baseline (fsst-rs Decompressor) vs OptimizedDecompressor
// ============================================================================

macro_rules! raw_bench_pair {
    ($baseline_name:ident, $optimized_name:ident, $data:expr) => {
        #[divan::bench]
        fn $baseline_name(bencher: Bencher) {
            let encoded = &*$data;
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

        #[divan::bench]
        fn $optimized_name(bencher: Bencher) {
            let encoded = &*$data;
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
    };
}

raw_bench_pair!(
    raw_baseline_short_strings,
    raw_optimized_short_strings,
    SHORT_STRINGS
);
raw_bench_pair!(
    raw_baseline_medium_strings,
    raw_optimized_medium_strings,
    MEDIUM_STRINGS
);
raw_bench_pair!(raw_baseline_emails, raw_optimized_emails, EMAILS);
raw_bench_pair!(
    raw_baseline_short_urls,
    raw_optimized_short_urls,
    SHORT_URLS
);
raw_bench_pair!(
    raw_baseline_clickbench_urls,
    raw_optimized_clickbench_urls,
    CLICKBENCH_URLS
);
raw_bench_pair!(raw_baseline_log_lines, raw_optimized_log_lines, LOG_LINES);
raw_bench_pair!(
    raw_baseline_json_strings,
    raw_optimized_json_strings,
    JSON_STRINGS
);
raw_bench_pair!(
    raw_baseline_file_paths,
    raw_optimized_file_paths,
    FILE_PATHS
);
