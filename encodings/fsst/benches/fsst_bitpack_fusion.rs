// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fusing bit-unpacking with FSST decompression into a single kernel.
//!
//! An FSST-compressed string column stores, per element, the byte length of its compressed code
//! sequence. To slice the code heap during decode we need a running offset, which means reading
//! those per-element lengths. Lengths are small and bounded, so they are the natural quantity to
//! bit-pack (cumulative offsets grow without bound and need a wide bit width).
//!
//! This benchmark explores fusing the two steps so the lengths never round-trip through memory.
//! Only the code child (bit-packed lengths + FSST decompress) is fused; the "other child"
//! (uncompressed lengths used to build views) is assumed already canonical and is not touched.
//!
//! Kernels compared:
//!
//! * `stream_fused` — **no tiles**: a single streaming loop holds a `u64` bit-accumulator in a
//!   register, refills it from a *contiguous* bit-packed stream, peels off the next `W`-bit
//!   length, and immediately FSST-decodes that one string. Nothing is staged in memory.
//! * `stream_unfused` — "bit-unpack, then FSST decode" over the same contiguous layout: pass 1
//!   extracts every length into a materialized `Vec<u32>`; pass 2 decodes from it.
//! * `tile_fused` — fuses FastLanes (transposed, 1024-element) unpacking with decode. FastLanes
//!   emits a whole 1024 block in strided lane order, so the lengths must be staged in a
//!   `[u32; 1024]` stack buffer before they can be walked sequentially for decode.
//! * `bulk` — a single `decompress_into` over the entire concatenated heap, the ceiling the
//!   production canonicalize path hits. Included for context; it produces no per-element
//!   boundaries so it cannot feed a step that needs them.
//!
//! Type is fixed to `u32` lengths. Common bit widths: `W = 8`, `12`, `16`. The kernels are
//! `#[inline(never)]` so they can be inspected with
//! `cargo asm -p vortex-fsst --bench fsst_bitpack_fusion <symbol>`.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;

use divan::Bencher;
use fastlanes::BitPacking;
use fsst::Compressor;
use fsst::Decompressor;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

fn main() {
    verify();
    divan::main();
}

const TILE: usize = 1024;

/// Number of strings to decode. All values are multiples of `TILE` so every FastLanes tile is
/// full and the tiled kernel stays branch-free at the block boundary.
const NUM_STRINGS: &[usize] = &[64 * TILE, 256 * TILE];

// ---------------------------------------------------------------------------------------------
// Streaming (tile-free) kernels over a contiguous LSB-first bit-packed length stream.
// ---------------------------------------------------------------------------------------------

/// Fused, tile-free: keep a `u64` accumulator in a register, refill from `packed`, peel one
/// `W`-bit length, and FSST-decode that string immediately. `packed` must have >= 16 bytes of
/// trailing padding so the byte-wise refill never reads out of bounds.
#[inline(never)]
fn stream_fused<const W: usize>(
    packed: &[u8],
    codes: &[u8],
    decompressor: &Decompressor<'_>,
    out: &mut [MaybeUninit<u8>],
    n: usize,
) -> usize {
    let mask = (1u64 << W) - 1;
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut bptr = 0usize;
    let mut code_pos = 0usize;
    let mut out_pos = 0usize;
    for _ in 0..n {
        if (nbits as usize) < W {
            while nbits <= 56 {
                acc |= (packed[bptr] as u64) << nbits;
                bptr += 1;
                nbits += 8;
            }
        }
        let clen = (acc & mask) as usize;
        acc >>= W;
        nbits -= W as u32;

        let m =
            decompressor.decompress_into(&codes[code_pos..code_pos + clen], &mut out[out_pos..]);
        code_pos += clen;
        out_pos += m;
    }
    out_pos
}

/// Unfused baseline over the same contiguous layout: materialize all lengths, then decode.
#[inline(never)]
fn stream_unfused<const W: usize>(
    packed: &[u8],
    codes: &[u8],
    decompressor: &Decompressor<'_>,
    out: &mut [MaybeUninit<u8>],
    n: usize,
) -> usize {
    // Pass 1: extract every length into a contiguous array.
    let mut lens = vec![0u32; n];
    let mask = (1u64 << W) - 1;
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    let mut bptr = 0usize;
    for slot in lens.iter_mut() {
        if (nbits as usize) < W {
            while nbits <= 56 {
                acc |= (packed[bptr] as u64) << nbits;
                bptr += 1;
                nbits += 8;
            }
        }
        *slot = (acc & mask) as u32;
        acc >>= W;
        nbits -= W as u32;
    }

    // Pass 2: FSST-decode each string using the materialized lengths.
    let mut code_pos = 0usize;
    let mut out_pos = 0usize;
    for &clen in &lens {
        let clen = clen as usize;
        let m =
            decompressor.decompress_into(&codes[code_pos..code_pos + clen], &mut out[out_pos..]);
        code_pos += clen;
        out_pos += m;
    }
    out_pos
}

// ---------------------------------------------------------------------------------------------
// FastLanes (transposed, 1024-element tile) kernel for comparison.
// ---------------------------------------------------------------------------------------------

/// Unpack a single full FastLanes tile of `1024` `u32` values packed at a compile-time width.
#[inline(never)]
fn unpack_tile<const W: usize, const B: usize>(packed: &[u32], out: &mut [u32; TILE]) {
    let input: &[u32; B] = packed[..B].try_into().unwrap();
    <u32 as BitPacking>::unpack::<W, B>(input, out);
}

/// Fused over FastLanes: per block, unpack the 1024 lengths into a stack tile, then decode them.
#[inline(never)]
fn tile_fused<const W: usize, const B: usize>(
    packed_lens: &[u32],
    codes: &[u8],
    decompressor: &Decompressor<'_>,
    out: &mut [MaybeUninit<u8>],
    n_tiles: usize,
) -> usize {
    let mut lens = [0u32; TILE];
    let mut code_pos = 0usize;
    let mut out_pos = 0usize;
    for t in 0..n_tiles {
        unpack_tile::<W, B>(&packed_lens[t * B..], &mut lens);
        for &clen in lens.iter() {
            let clen = clen as usize;
            let m = decompressor
                .decompress_into(&codes[code_pos..code_pos + clen], &mut out[out_pos..]);
            code_pos += clen;
            out_pos += m;
        }
    }
    out_pos
}

/// Reference: decode the whole concatenated heap in one FSST call (no per-element boundaries).
#[inline(never)]
fn bulk_decode(codes: &[u8], decompressor: &Decompressor<'_>, out: &mut [MaybeUninit<u8>]) -> usize {
    decompressor.decompress_into(codes, out)
}

// ---------------------------------------------------------------------------------------------
// Fixture.
// ---------------------------------------------------------------------------------------------

struct Fixture {
    codes: Vec<u8>,
    /// Lengths packed contiguously (LSB-first) at W = 8 / 12 / 16, each with trailing padding.
    contig8: Vec<u8>,
    contig12: Vec<u8>,
    contig16: Vec<u8>,
    /// Lengths packed with FastLanes at W = 8 (B = 256) and W = 16 (B = 512).
    fl8: Vec<u32>,
    fl16: Vec<u32>,
    compressor: Compressor,
    n: usize,
    n_tiles: usize,
    /// Total decoded size plus slack for FSST's 8-byte symbol stores.
    out_cap: usize,
    /// Expected decoded bytes, for correctness checks.
    expected: Vec<u8>,
}

impl Fixture {
    fn decompressor(&self) -> Decompressor<'_> {
        self.compressor.decompressor()
    }
}

/// Build `n` compressible strings, FSST-compress them, and bit-pack the per-string compressed
/// lengths both contiguously and with FastLanes. `n` must be a multiple of `TILE`.
fn build_fixture(n: usize) -> Fixture {
    assert_eq!(n % TILE, 0, "n must be a multiple of {TILE}");
    let mut rng = StdRng::seed_from_u64(0);

    // A small token vocabulary keeps the strings compressible and keeps every compressed length
    // below 256 so it fits in 8 bits.
    let tokens: &[&[u8]] = &[
        b"http://", b"https://", b"www.", b".com/", b"index", b"page", b"query=", b"&id=",
        b"vortex", b"data", b"/path/to/", b"resource", b"-2024-", b"value", b"_field",
    ];
    let strings: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            let parts = rng.random_range(2..6);
            let mut s = Vec::new();
            for _ in 0..parts {
                s.extend_from_slice(tokens.choose(&mut rng).unwrap());
            }
            s
        })
        .collect();

    let refs: Vec<&[u8]> = strings.iter().map(Vec::as_slice).collect();
    let compressor = Compressor::train(&refs);

    let mut codes = Vec::new();
    let mut lens = vec![0u32; n];
    let mut expected = Vec::new();
    for (i, s) in strings.iter().enumerate() {
        let c = compressor.compress(s);
        assert!(
            c.len() < 256,
            "compressed length {} does not fit in 8 bits",
            c.len()
        );
        lens[i] = c.len() as u32;
        codes.extend_from_slice(&c);
        expected.extend_from_slice(s);
    }

    let n_tiles = n / TILE;
    Fixture {
        codes,
        contig8: pack_contiguous(&lens, 8),
        contig12: pack_contiguous(&lens, 12),
        contig16: pack_contiguous(&lens, 16),
        fl8: pack_fastlanes(&lens, 8, n_tiles),
        fl16: pack_fastlanes(&lens, 16, n_tiles),
        compressor,
        n,
        n_tiles,
        out_cap: expected.len() + 16,
        expected,
    }
}

/// Pack `lens` into a contiguous LSB-first bit stream of `width`-bit values, with 16 trailing
/// padding bytes so the streaming refill can over-read safely.
fn pack_contiguous(lens: &[u32], width: usize) -> Vec<u8> {
    let mask = (1u64 << width) - 1;
    let mut out = Vec::with_capacity((lens.len() * width).div_ceil(8) + 16);
    let mut acc: u64 = 0;
    let mut nbits = 0u32;
    for &v in lens {
        acc |= (v as u64 & mask) << nbits;
        nbits += width as u32;
        while nbits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push(acc as u8);
    }
    out.extend_from_slice(&[0u8; 16]);
    out
}

/// Bit-pack `lens` (length = `n_tiles * TILE`) with FastLanes at `width`.
fn pack_fastlanes(lens: &[u32], width: usize, n_tiles: usize) -> Vec<u32> {
    let words_per_tile = 128 * width / size_of::<u32>();
    let mut packed = vec![0u32; n_tiles * words_per_tile];
    for t in 0..n_tiles {
        let src: &[u32; TILE] = lens[t * TILE..][..TILE].try_into().unwrap();
        let dst = &mut packed[t * words_per_tile..][..words_per_tile];
        // SAFETY: src is exactly 1024 elements, dst is exactly 128 * width / 32 words.
        unsafe { <u32 as BitPacking>::unchecked_pack(width, src, dst) };
    }
    packed
}

// ---------------------------------------------------------------------------------------------
// Benches.
// ---------------------------------------------------------------------------------------------

macro_rules! stream_bench {
    ($fused:ident, $unfused:ident, $w:literal, $contig:ident) => {
        #[divan::bench(args = NUM_STRINGS)]
        fn $fused(bencher: Bencher, n: usize) {
            let fx = build_fixture(n);
            bencher
                .with_inputs(|| Vec::<u8>::with_capacity(fx.out_cap))
                .bench_refs(|out| {
                    stream_fused::<$w>(
                        &fx.$contig,
                        &fx.codes,
                        &fx.decompressor(),
                        out.spare_capacity_mut(),
                        fx.n,
                    )
                });
        }

        #[divan::bench(args = NUM_STRINGS)]
        fn $unfused(bencher: Bencher, n: usize) {
            let fx = build_fixture(n);
            bencher
                .with_inputs(|| Vec::<u8>::with_capacity(fx.out_cap))
                .bench_refs(|out| {
                    stream_unfused::<$w>(
                        &fx.$contig,
                        &fx.codes,
                        &fx.decompressor(),
                        out.spare_capacity_mut(),
                        fx.n,
                    )
                });
        }
    };
}

stream_bench!(stream_fused_w8, stream_unfused_w8, 8, contig8);
stream_bench!(stream_fused_w12, stream_unfused_w12, 12, contig12);
stream_bench!(stream_fused_w16, stream_unfused_w16, 16, contig16);

#[divan::bench(args = NUM_STRINGS)]
fn tile_fused_w8(bencher: Bencher, n: usize) {
    let fx = build_fixture(n);
    bencher
        .with_inputs(|| Vec::<u8>::with_capacity(fx.out_cap))
        .bench_refs(|out| {
            tile_fused::<8, 256>(
                &fx.fl8,
                &fx.codes,
                &fx.decompressor(),
                out.spare_capacity_mut(),
                fx.n_tiles,
            )
        });
}

#[divan::bench(args = NUM_STRINGS)]
fn tile_fused_w16(bencher: Bencher, n: usize) {
    let fx = build_fixture(n);
    bencher
        .with_inputs(|| Vec::<u8>::with_capacity(fx.out_cap))
        .bench_refs(|out| {
            tile_fused::<16, 512>(
                &fx.fl16,
                &fx.codes,
                &fx.decompressor(),
                out.spare_capacity_mut(),
                fx.n_tiles,
            )
        });
}

#[divan::bench(args = NUM_STRINGS)]
fn bulk(bencher: Bencher, n: usize) {
    let fx = build_fixture(n);
    bencher
        .with_inputs(|| Vec::<u8>::with_capacity(fx.out_cap))
        .bench_refs(|out| bulk_decode(&fx.codes, &fx.decompressor(), out.spare_capacity_mut()));
}

/// Decode with every kernel and assert they all reproduce the original bytes. Runs once at
/// startup (the bench harness is `harness = false`, so a `#[test]` would never be invoked).
fn verify() {
    let fx = build_fixture(2 * TILE);
    let dec = fx.decompressor();

    let run = |label: &str, n: usize, mut out: Vec<u8>| {
        // SAFETY: kernels wrote `n` initialized bytes into the spare capacity.
        unsafe { out.set_len(n) };
        assert!(out == fx.expected, "{label} produced wrong bytes");
    };

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = stream_fused::<8>(&fx.contig8, &fx.codes, &dec, o.spare_capacity_mut(), fx.n);
    run("stream_fused W=8", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = stream_fused::<12>(&fx.contig12, &fx.codes, &dec, o.spare_capacity_mut(), fx.n);
    run("stream_fused W=12", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = stream_fused::<16>(&fx.contig16, &fx.codes, &dec, o.spare_capacity_mut(), fx.n);
    run("stream_fused W=16", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = stream_unfused::<12>(&fx.contig12, &fx.codes, &dec, o.spare_capacity_mut(), fx.n);
    run("stream_unfused W=12", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = tile_fused::<8, 256>(&fx.fl8, &fx.codes, &dec, o.spare_capacity_mut(), fx.n_tiles);
    run("tile_fused W=8", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = tile_fused::<16, 512>(&fx.fl16, &fx.codes, &dec, o.spare_capacity_mut(), fx.n_tiles);
    run("tile_fused W=16", n, o);

    let mut o = Vec::<u8>::with_capacity(fx.out_cap);
    let n = bulk_decode(&fx.codes, &dec, o.spare_capacity_mut());
    run("bulk", n, o);
}
