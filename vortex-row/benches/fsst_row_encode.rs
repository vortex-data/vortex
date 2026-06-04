// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

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
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
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

fn main() {
    divan::main();
}

/// Status quo: decompress FSST to a canonical `VarBinView`, then row-encode it.
#[divan::bench]
fn fsst_unpack_then_convert(bencher: divan::Bencher) {
    let (fsst, total_bytes) = build_fsst();
    let encoder = RowEncoder::default();
    bencher.counter(BytesCount::new(total_bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decoded = fsst.clone().execute::<Canonical>(&mut ctx).unwrap().into_array();
        encoder.encode(&[decoded], &mut ctx).unwrap()
    });
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
    bencher.counter(BytesCount::new(total_bytes)).bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        encoder.encode(std::slice::from_ref(&decoded), &mut ctx).unwrap()
    });
}
