// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use fsst::CompressorBuilder;
use fsst::Symbol;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_buffer::ByteBufferMut;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_compress_iter;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

// --- Decompress component isolation benchmarks ---
// These benchmarks decompose the full FSST decompress pipeline into its parts:
//   raw_decompress_only: just the fsst-rs decompressor (no allocation, no view building)
//   view_build_only: just the view construction loop (no decompression)

const COMPONENT_ARGS: &[(usize, usize, u8)] = &[(10_000, 16, 4), (10_000, 64, 4), (10_000, 256, 4)];

fn build_single_fsst(string_count: usize, avg_len: usize, unique_chars: u8) -> FSSTArray {
    let mut rng = StdRng::seed_from_u64(42);
    let strings: Vec<Option<Box<[u8]>>> = (0..string_count)
        .map(|_| {
            let len = avg_len * rng.random_range(80..=120) / 100;
            let s: Vec<u8> = (0..len)
                .map(|_| b'a' + rng.random_range(0..unique_chars))
                .collect();
            Some(s.into_boxed_slice())
        })
        .collect();
    let array = VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable));
    let compressor = fsst_train_compressor(&array);
    fsst_compress(array, &compressor)
}

#[divan::bench(args = COMPONENT_ARGS)]
fn raw_decompress_only(
    bencher: Bencher,
    &(string_count, avg_len, unique_chars): &(usize, usize, u8),
) {
    let encoded = build_single_fsst(string_count, avg_len, unique_chars);
    let compressed = encoded.codes().sliced_bytes();
    let decompressor = encoded.decompressor();
    let lens = encoded
        .uncompressed_lengths()
        .as_opt::<Primitive>()
        .unwrap();
    #[allow(clippy::cast_sign_loss)]
    let total_size: usize = lens.as_slice::<i32>().iter().map(|&x| x as usize).sum();

    bencher
        .with_inputs(|| ByteBufferMut::with_capacity(total_size + 7))
        .bench_refs(|buf| {
            let len = decompressor.decompress_into(compressed.as_slice(), buf.spare_capacity_mut());
            unsafe { buf.set_len(len) };
        })
}

#[divan::bench(args = COMPONENT_ARGS)]
fn view_build_only(bencher: Bencher, &(string_count, avg_len, unique_chars): &(usize, usize, u8)) {
    let encoded = build_single_fsst(string_count, avg_len, unique_chars);
    let compressed = encoded.codes().sliced_bytes();
    let decompressor = encoded.decompressor();
    let lens = encoded
        .uncompressed_lengths()
        .as_opt::<Primitive>()
        .unwrap();
    let lens_slice = lens.as_slice::<i32>();
    #[allow(clippy::cast_sign_loss)]
    let total_size: usize = lens_slice.iter().map(|&x| x as usize).sum();

    let mut buf = ByteBufferMut::with_capacity(total_size + 7);
    let len = decompressor.decompress_into(compressed.as_slice(), buf.spare_capacity_mut());
    unsafe { buf.set_len(len) };

    bencher
        .with_inputs(|| buf.clone())
        .bench_refs(|buf| build_views(0, MAX_BUFFER_LEN, std::mem::take(buf), lens_slice))
}

fn main() {
    warm_up_vtables();
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

// --- Escape code ratio benchmarks ---
// Tests decompression speed with varying proportions of escape codes in the symbol table.
// Few escape codes = most codes map to multi-byte symbols (good compression).
// Many escape codes = most bytes are escaped (poor compression, more work per byte).

/// Build an FSST array with a controlled escape code ratio.
/// `escape_fraction` controls what fraction of the input bytes will NOT be in the symbol table.
fn build_fsst_with_escape_ratio(
    string_count: usize,
    avg_len: usize,
    escape_fraction: f64,
) -> FSSTArray {
    let mut rng = StdRng::seed_from_u64(42);

    // Characters that will be trained into symbols (a-d)
    let symbol_chars: Vec<u8> = (b'a'..=b'd').collect();
    // Characters that will be escaped (rare, not in training data)
    let escape_chars: Vec<u8> = (0xF0..=0xFE).collect();

    let mut strings = Vec::with_capacity(string_count);
    for _ in 0..string_count {
        let len = avg_len * rng.random_range(80..=120) / 100;
        let s: Vec<u8> = (0..len)
            .map(|_| {
                if rng.random_bool(escape_fraction) {
                    escape_chars[rng.random_range(0..escape_chars.len())]
                } else {
                    symbol_chars[rng.random_range(0..symbol_chars.len())]
                }
            })
            .collect();
        strings.push(Some(s));
    }

    let array = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&array);
    fsst_compress(array, &compressor)
}

// (string_count, avg_len, escape_fraction)
const ESCAPE_RATIO_ARGS: &[(usize, usize, u8)] = &[
    // No escapes: all bytes are in the symbol table
    (10_000, 64, 0),
    // 10% escapes
    (10_000, 64, 10),
    // 30% escapes
    (10_000, 64, 30),
    // 50% escapes
    (10_000, 64, 50),
    // 80% escapes: most bytes are escaped
    (10_000, 64, 80),
    // 100% escapes: symbol table is useless
    (10_000, 64, 100),
];

#[divan::bench(args = ESCAPE_RATIO_ARGS)]
fn decompress_escape_ratio(
    bencher: Bencher,
    &(string_count, avg_len, escape_pct): &(usize, usize, u8),
) {
    let escape_frac = escape_pct as f64 / 100.0;
    let encoded = build_fsst_with_escape_ratio(string_count, avg_len, escape_frac);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical())
}

// --- Symbol length benchmarks ---
// Tests decompression with symbol tables containing short (1-2 byte) vs long (6-8 byte) symbols.

/// Build an FSST array where the symbol table is trained on data with specific patterns
/// that produce short or long symbols.
fn build_fsst_with_symbol_lengths(string_count: usize, avg_len: usize, long: bool) -> FSSTArray {
    let mut rng = StdRng::seed_from_u64(42);

    let mut strings = Vec::with_capacity(string_count);
    if long {
        // Use repeating 8-byte patterns to encourage long symbols
        let patterns: &[&[u8]] = &[b"abcdefgh", b"ijklmnop", b"qrstuvwx", b"yzABCDEF"];
        for _ in 0..string_count {
            let len = avg_len * rng.random_range(80..=120) / 100;
            let mut s = Vec::with_capacity(len);
            while s.len() < len {
                let pat = patterns[rng.random_range(0..patterns.len())];
                let remaining = len - s.len();
                s.extend_from_slice(&pat[..remaining.min(pat.len())]);
            }
            strings.push(Some(s));
        }
    } else {
        // Use highly varied single bytes to produce 1-byte symbols
        for _ in 0..string_count {
            let len = avg_len * rng.random_range(80..=120) / 100;
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect();
            strings.push(Some(s));
        }
    }

    let array = VarBinArray::from_iter(
        strings
            .into_iter()
            .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
        DType::Binary(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&array);
    fsst_compress(array, &compressor)
}

// (string_count, avg_len, long_symbols)
const SYMBOL_LENGTH_ARGS: &[(usize, usize, bool)] = &[
    // Short symbols (1-2 byte), small strings
    (10_000, 16, false),
    // Short symbols, medium strings
    (10_000, 64, false),
    // Short symbols, large strings
    (10_000, 256, false),
    // Long symbols (6-8 byte), small strings
    (10_000, 16, true),
    // Long symbols, medium strings
    (10_000, 64, true),
    // Long symbols, large strings
    (10_000, 256, true),
];

#[divan::bench(args = SYMBOL_LENGTH_ARGS)]
fn decompress_symbol_length(
    bencher: Bencher,
    &(string_count, avg_len, long_symbols): &(usize, usize, bool),
) {
    let encoded = build_fsst_with_symbol_lengths(string_count, avg_len, long_symbols);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical())
}

// --- Pre-constructed FSST array benchmarks ---
// Tests with manually constructed symbol tables (known escape code count).

fn build_fsst_manual_symbols(string_count: usize, avg_len: usize, n_symbols: usize) -> FSSTArray {
    let mut builder = CompressorBuilder::new();
    for i in 0..n_symbols.min(255) {
        #[allow(clippy::cast_possible_truncation)]
        let byte = i as u8;
        let mut sym_bytes = [0u8; 8];
        // Create 2-byte symbols from pairs
        sym_bytes[0] = b'a' + (byte % 26);
        sym_bytes[1] = b'a' + ((byte / 26) % 26);
        builder.insert(Symbol::from_slice(&sym_bytes), 2);
    }
    let compressor = builder.build();

    let mut rng = StdRng::seed_from_u64(42);
    let mut strings: Vec<Vec<u8>> = Vec::with_capacity(string_count);
    for _ in 0..string_count {
        let len = avg_len * rng.random_range(80..=120) / 100;
        let s: Vec<u8> = (0..len).map(|_| b'a' + rng.random_range(0..26u8)).collect();
        strings.push(s);
    }

    let refs: Vec<Option<&[u8]>> = strings.iter().map(|s| Some(s.as_slice())).collect();
    fsst_compress_iter(
        refs.into_iter(),
        string_count,
        DType::Binary(Nullability::NonNullable),
        &compressor,
    )
}

// (string_count, avg_len, n_symbols)
const SYMBOL_COUNT_ARGS: &[(usize, usize, usize)] = &[
    // Very few symbols (more escapes)
    (10_000, 64, 10),
    // Moderate symbols
    (10_000, 64, 100),
    // Full symbol table
    (10_000, 64, 255),
];

#[divan::bench(args = SYMBOL_COUNT_ARGS)]
fn decompress_symbol_count(
    bencher: Bencher,
    &(string_count, avg_len, n_symbols): &(usize, usize, usize),
) {
    let encoded = build_fsst_manual_symbols(string_count, avg_len, n_symbols);

    bencher
        .with_inputs(|| &encoded)
        .bench_refs(|encoded| encoded.to_canonical())
}

// --- Chunked decompress benchmarks ---
// Tests with varying chunk counts and sizes.

fn build_chunked_fsst(
    n_chunks: usize,
    strings_per_chunk: usize,
    avg_len: usize,
    unique_chars: u8,
) -> ChunkedArray {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n_chunks)
        .map(|_| {
            let mut strings = Vec::with_capacity(strings_per_chunk);
            for _ in 0..strings_per_chunk {
                let len = avg_len * rng.random_range(80..=120) / 100;
                let s: Vec<u8> = (0..len)
                    .map(|_| b'a' + rng.random_range(0..unique_chars))
                    .collect();
                strings.push(Some(s.into_boxed_slice()));
            }
            let array = VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable));
            let compressor = fsst_train_compressor(&array);
            fsst_compress(array, &compressor).into_array()
        })
        .collect::<ChunkedArray>()
}

// (n_chunks, strings_per_chunk, avg_len, unique_chars)
const CHUNKED_DECOMPRESS_ARGS: &[(usize, usize, usize, u8)] = &[
    // Few large chunks
    (5, 10_000, 64, 4),
    (5, 10_000, 64, 16),
    // Many small chunks
    (100, 500, 64, 4),
    (100, 500, 64, 16),
    // Many tiny chunks (high per-chunk overhead)
    (1000, 50, 16, 4),
    (1000, 50, 64, 4),
];

#[divan::bench(args = CHUNKED_DECOMPRESS_ARGS)]
fn chunked_decompress_to_canonical(
    bencher: Bencher,
    &(n_chunks, strings_per_chunk, avg_len, unique_chars): &(usize, usize, usize, u8),
) {
    let array = build_chunked_fsst(n_chunks, strings_per_chunk, avg_len, unique_chars);

    bencher
        .with_inputs(|| &array)
        .bench_refs(|array| array.to_canonical())
}

#[divan::bench(args = CHUNKED_DECOMPRESS_ARGS)]
fn chunked_decompress_to_builder(
    bencher: Bencher,
    &(n_chunks, strings_per_chunk, avg_len, unique_chars): &(usize, usize, usize, u8),
) {
    let array = build_chunked_fsst(n_chunks, strings_per_chunk, avg_len, unique_chars);

    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity(
                DType::Binary(Nullability::NonNullable),
                array.len(),
            );
            array.append_to_builder(&mut builder, ctx).unwrap();
            builder.finish()
        })
}

#[divan::bench(args = CHUNKED_DECOMPRESS_ARGS)]
fn chunked_decompress_batch(
    bencher: Bencher,
    &(n_chunks, strings_per_chunk, avg_len, unique_chars): &(usize, usize, usize, u8),
) {
    let array = build_chunked_fsst(n_chunks, strings_per_chunk, avg_len, unique_chars);

    // Pre-collect FSST chunks for batch decode
    let fsst_chunks: Vec<&FSSTArray> = array
        .chunks()
        .iter()
        .map(|c| c.as_opt::<vortex_fsst::FSST>().unwrap())
        .collect();

    bencher
        .with_inputs(|| (&fsst_chunks, SESSION.create_execution_ctx()))
        .bench_refs(|(chunks, ctx)| vortex_fsst::fsst_batch_decode(chunks, ctx).unwrap())
}
