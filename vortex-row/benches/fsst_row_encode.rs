// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

//! Row-encoding an FSST-compressed string column: the only realizable strategy is
//! "unpack then convert" (decompress FSST to a canonical `VarBinView`, then row-encode it),
//! because FSST is **not order-preserving** — its 1-byte codes are assigned by compression
//! gain, not by value, so the compressed bytes cannot be compared lexicographically. A
//! hypothetical "direct" kernel could only *fuse* decompression with row-key emission; it
//! still has to expand every symbol.
//!
//! These benchmarks measure the full path and its two phases so the fusion opportunity is
//! quantifiable:
//!   * `fsst_unpack_then_convert` — decompress + row-encode (the status quo).
//!   * `fsst_decompress_only`     — decompress alone (the irreducible floor: a direct kernel
//!     must still produce these bytes).
//!   * `plain_row_encode_only`    — row-encode an already-decompressed `VarBinView` (the part
//!     a fused kernel would overlap with decompression; its writes into the intermediate
//!     buffer + views are what fusion removes).

use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_row::RowEncoder;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const N: usize = 100_000;
const AVG_LEN: usize = 64;
const UNIQUE_CHARS: u8 = 8;

/// Generate compressible, multi-block (>32 byte) strings over a small alphabet so FSST finds
/// a strong symbol table — the regime where a direct kernel would matter most.
fn generate_strings() -> (VarBinArray, u64) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut strings = Vec::with_capacity(N);
    let mut total_bytes: u64 = 0;
    for _ in 0..N {
        let len = AVG_LEN * rng.random_range(50..=150) / 100;
        total_bytes += len as u64;
        let s = (0..len)
            .map(|_| rng.random_range(b'a'..(b'a' + UNIQUE_CHARS)) as char)
            .collect::<String>()
            .into_bytes();
        strings.push(Some(s.into_boxed_slice()));
    }
    let arr = VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable));
    (arr, total_bytes)
}

fn build_fsst() -> (ArrayRef, u64) {
    let (arr, total_bytes) = generate_strings();
    let compressor = fsst_train_compressor(&arr);
    let len = arr.len();
    let dtype = arr.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let fsst = fsst_compress(arr, len, &dtype, &compressor, &mut ctx).into_array();
    (fsst, total_bytes)
}

fn decompress(fsst: &ArrayRef) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    fsst.clone()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

const VARLEN_BLOCK: usize = 32;
const VARLEN_BLOCK_TOTAL: usize = 33;
// Sentinel for a non-empty varlen value (ascending, non-null) — value is irrelevant to timing.
const NON_EMPTY_SENTINEL: u8 = 0x02;

/// Encoded row-key length for a non-empty value of `len` decompressed bytes: a leading
/// sentinel plus `ceil(len/32)` 32-byte blocks, each followed by a continuation/length byte.
fn encoded_len(len: usize) -> u32 {
    if len == 0 {
        1
    } else {
        1 + (len.div_ceil(VARLEN_BLOCK) as u32) * VARLEN_BLOCK_TOTAL as u32
    }
}

/// Block-encode `bytes` (ascending) into `out`, matching vortex-row's varlen body format.
fn block_encode(bytes: &[u8], out: &mut [u8]) {
    let len = bytes.len();
    let full = len / VARLEN_BLOCK;
    let partial = len % VARLEN_BLOCK;
    let (full_to_write, partial_len) = if partial == 0 {
        (full - 1, VARLEN_BLOCK)
    } else {
        (full, partial)
    };
    let mut src = 0;
    let mut dst = 0;
    for _ in 0..full_to_write {
        out[dst..dst + VARLEN_BLOCK].copy_from_slice(&bytes[src..src + VARLEN_BLOCK]);
        out[dst + VARLEN_BLOCK] = 0xFF;
        src += VARLEN_BLOCK;
        dst += VARLEN_BLOCK_TOTAL;
    }
    out[dst..dst + partial_len].copy_from_slice(&bytes[src..src + partial_len]);
    for b in &mut out[dst + partial_len..dst + VARLEN_BLOCK] {
        *b = 0;
    }
    out[dst + VARLEN_BLOCK] = partial_len as u8;
}

/// Fused FSST → row-key kernel: bulk-decompress the code heap into one contiguous buffer (no
/// intermediate `VarBinViewArray`), then block-encode each row straight into the row-key
/// `ListView<u8>` using the stored `uncompressed_lengths` for boundaries (no size-pass walk).
fn fast_fused(fsst: &ArrayRef) -> ArrayRef {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = fsst.as_opt::<FSST>().expect("FSST array");

    // Per-row decompressed lengths are already stored — the size pass is free.
    let lens_arr = view
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let lens: Vec<usize> = match_each_integer_ptype!(lens_arr.ptype(), |P| {
        lens_arr
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .collect()
    });

    // Bulk-decompress the whole code heap once into a contiguous buffer (no VarBinView).
    let heap = view.codes_bytes();
    let total: usize = lens.iter().sum();
    let decompressor = view.decompressor();
    let mut decompressed = ByteBufferMut::with_capacity(total + 7);
    let n = decompressor.decompress_into(heap.as_slice(), decompressed.spare_capacity_mut());
    unsafe { decompressed.set_len(n) };
    let bytes = decompressed.as_slice();

    // Size + offsets for the row-key ListView (lengths are free, no view walk).
    let nrows = lens.len();
    let mut offsets: Vec<u32> = Vec::with_capacity(nrows);
    let mut sizes: Vec<u32> = Vec::with_capacity(nrows);
    let mut acc: u32 = 0;
    for &l in &lens {
        offsets.push(acc);
        let sz = encoded_len(l);
        sizes.push(sz);
        acc += sz;
    }

    // Block-encode every row directly into the elements buffer. No zero-init (every byte is
    // written: sentinel + block body with zero-padded final block) and no Vec→Buffer copy.
    let mut out = ByteBufferMut::with_capacity(acc as usize);
    unsafe { out.set_len(acc as usize) };
    let out_slice = out.as_mut_slice();
    let mut src = 0usize;
    for (i, &l) in lens.iter().enumerate() {
        let pos = offsets[i] as usize;
        out_slice[pos] = NON_EMPTY_SENTINEL;
        if l != 0 {
            block_encode(&bytes[src..src + l], &mut out_slice[pos + 1..]);
        }
        src += l;
    }

    let elements = PrimitiveArray::new(out.freeze(), Validity::NonNullable);
    let offsets_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&offsets), Validity::NonNullable);
    let sizes_arr = PrimitiveArray::new(Buffer::<u32>::copy_from(&sizes), Validity::NonNullable);
    ListViewArray::try_new(
        elements.into_array(),
        offsets_arr.into_array(),
        sizes_arr.into_array(),
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

/// "Scatter right": keep FSST's fast contiguous bulk decompressor, but run it into a
/// cache-resident scratch one row-batch at a time, then scatter each row into block form from
/// cache. The decompressed bytes never round-trip through main memory — unlike `fast_fused`,
/// which materializes the whole 6.4 MB decompressed buffer and reads it back to block-encode.
fn fast_scatter(fsst: &ArrayRef) -> ArrayRef {
    // Scratch sized to stay resident in L1/L2; each batch decompresses up to this many bytes.
    const SCRATCH: usize = 16 * 1024;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let view = fsst.as_opt::<FSST>().expect("FSST array");

    let lens_arr = view
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let lens: Vec<usize> = match_each_integer_ptype!(lens_arr.ptype(), |P| {
        lens_arr
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .collect()
    });
    let nrows = lens.len();

    // Per-row compressed code offsets (relative to the sliced heap start).
    let codes = view.codes();
    let heap = codes.sliced_bytes();
    let code_off_arr = codes
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let base = match_each_integer_ptype!(code_off_arr.ptype(), |P| {
        code_off_arr.as_slice::<P>()[0] as usize
    });
    let code_off: Vec<usize> = match_each_integer_ptype!(code_off_arr.ptype(), |P| {
        code_off_arr
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize - base)
            .collect()
    });

    // Output sizing (free from stored lengths).
    let mut offsets: Vec<u32> = Vec::with_capacity(nrows);
    let mut sizes: Vec<u32> = Vec::with_capacity(nrows);
    let mut acc: u32 = 0;
    let mut max_row = 0usize;
    for &l in &lens {
        offsets.push(acc);
        let sz = encoded_len(l);
        sizes.push(sz);
        acc += sz;
        max_row = max_row.max(l);
    }
    let mut out = ByteBufferMut::with_capacity(acc as usize);
    unsafe { out.set_len(acc as usize) };
    let out_slice = out.as_mut_slice();

    let decompressor = view.decompressor();
    let scratch_cap = SCRATCH.max(max_row) + 8;
    let mut scratch = ByteBufferMut::with_capacity(scratch_cap);

    let mut r = 0usize;
    while r < nrows {
        // Grow a batch until it would overflow the scratch (always at least one row).
        let bs = r;
        let mut batch_bytes = 0usize;
        while r < nrows && (r == bs || batch_bytes + lens[r] <= SCRATCH) {
            batch_bytes += lens[r];
            r += 1;
        }
        let be = r;

        // Decompress this batch's codes in one fast call into the cache-resident scratch.
        let cslice = &heap.as_slice()[code_off[bs]..code_off[be]];
        let n = decompressor.decompress_into(cslice, scratch.spare_capacity_mut());
        unsafe { scratch.set_len(n) };
        let sbytes = scratch.as_slice();

        // Scatter each row from cache into block form.
        let mut local = 0usize;
        for i in bs..be {
            let l = lens[i];
            let pos = offsets[i] as usize;
            out_slice[pos] = NON_EMPTY_SENTINEL;
            if l != 0 {
                block_encode(&sbytes[local..local + l], &mut out_slice[pos + 1..]);
            }
            local += l;
        }
        unsafe { scratch.set_len(0) };
    }

    let elements = PrimitiveArray::new(out.freeze(), Validity::NonNullable);
    let offsets_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&offsets), Validity::NonNullable);
    let sizes_arr = PrimitiveArray::new(Buffer::<u32>::copy_from(&sizes), Validity::NonNullable);
    ListViewArray::try_new(
        elements.into_array(),
        offsets_arr.into_array(),
        sizes_arr.into_array(),
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

fn main() {
    // Correctness: the batched cache-resident scatter must produce identical row keys to the
    // straightforward fused path.
    {
        let (fsst, _) = build_fsst();
        assert_arrays_eq!(fast_scatter(&fsst), fast_fused(&fsst));
    }
    divan::main();
}

/// "Scatter right" fused path: cache-resident batched decompress + scatter into block form.
#[divan::bench]
fn fsst_fast_scatter(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench_local(|| fast_scatter(&fsst));
}

/// Status quo: decompress FSST to a canonical `VarBinView`, then row-encode it.
#[divan::bench]
fn fsst_unpack_then_convert(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    let encoder = RowEncoder::default();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench_local(|| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let decoded = fsst
                .clone()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
                .into_array();
            encoder.encode(&[decoded], &mut ctx).unwrap()
        });
}

/// Fused fast path: bulk-decompress directly into the row-key block format, skipping the
/// intermediate `VarBinViewArray` and the generic row-encoder (size pass is free).
#[divan::bench]
fn fsst_fast_fused(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench_local(|| fast_fused(&fsst));
}

/// Irreducible floor: FSST decompression alone (a direct kernel must still produce these
/// bytes, since the sort key *is* the decompressed bytes).
#[divan::bench]
fn fsst_decompress_only(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench_local(|| decompress(&fsst));
}

/// Row-encode an already-decompressed `VarBinView`. The writes into the decompressed buffer +
/// views that precede this step are what a fused direct kernel would eliminate.
#[divan::bench]
fn plain_row_encode_only(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    let decoded = decompress(&fsst);
    let encoder = RowEncoder::default();
    bencher
        .counter(BytesCount::new(total_bytes))
        .bench_local(|| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            encoder
                .encode(std::slice::from_ref(&decoded), &mut ctx)
                .unwrap()
        });
}
