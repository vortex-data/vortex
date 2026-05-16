// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bare-CPU bench: gather (take) vs SIMD compare (cmp), no Vortex/Arrow pipeline.
//!
//! Mirrors the inner kernel each path in `dict::compute::compare` runs:
//!   - take_bool: walks codes, indexed bit-read into a small bool dict → output bits
//!   - cmp: walks codes, compares each against scalar → output bits
//! Two output flavors each: `Vec<bool>` (one byte per result) and `BitBuffer` (one bit
//! per result, via `BitBufferMut`).
//!
//! Goal: confirm whether the gather pattern (`bools[codes[i]]`) actually costs more than
//! the SIMD-friendly `codes[i] < threshold` once the pipeline overhead is removed.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

fn make_codes(n: usize) -> Vec<u16> {
    // Cyclic but unpredictable enough that the compiler can't fold lookups.
    (0..n).map(|i| (i.wrapping_mul(2654435761) & 0x3FF) as u16).collect()
}

fn make_bool_dict() -> Vec<bool> {
    (0..1024).map(|i| (i * 31) % 7 == 0).collect()
}

fn make_bit_dict() -> BitBuffer {
    let mut bb = BitBufferMut::with_capacity(1024);
    for i in 0..1024 {
        bb.append((i * 31) % 7 == 0);
    }
    bb.freeze()
}

// ---------------------------------------------------------------------------
// take: bools[codes[i]] for i in 0..N
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn take_vec_to_vec(bencher: Bencher, n: usize) {
    let codes = make_codes(n);
    let bools = make_bool_dict();
    bencher.bench(|| {
        let mut out: Vec<bool> = Vec::with_capacity(n);
        for &c in &codes {
            // SAFETY: codes are masked to 0x3FF, dict has 1024 entries.
            unsafe { out.push(*bools.get_unchecked(c as usize)) }
        }
        black_box(out)
    });
}

#[divan::bench(args = SIZES)]
fn take_vec_to_bits(bencher: Bencher, n: usize) {
    let codes = make_codes(n);
    let bools = make_bool_dict();
    bencher.bench(|| {
        let bits = BitBuffer::collect_bool(n, |i| {
            // SAFETY: idx in 0..n, codes len == n; code masked to 0x3FF, dict has 1024 entries.
            unsafe { *bools.get_unchecked(*codes.get_unchecked(i) as usize) }
        });
        black_box(bits)
    });
}

#[divan::bench(args = SIZES)]
fn take_bits_to_bits(bencher: Bencher, n: usize) {
    // Bit-packed bool dict → bit-packed output. This is the exact pattern Vortex's
    // take_bool_impl uses (collect_bool + get_bit).
    use vortex_buffer::get_bit;
    let codes = make_codes(n);
    let dict = make_bit_dict();
    let dict_bytes = dict.inner().as_ref();
    let dict_off = dict.offset();
    bencher.bench(|| {
        let bits = BitBuffer::collect_bool(n, |i| {
            // SAFETY: same bounds as above.
            let idx = unsafe { *codes.get_unchecked(i) } as usize;
            get_bit(dict_bytes, dict_off + idx)
        });
        black_box(bits)
    });
}

// ---------------------------------------------------------------------------
// cmp: codes[i] < threshold for i in 0..N
// ---------------------------------------------------------------------------

#[divan::bench(args = SIZES)]
fn cmp_to_vec(bencher: Bencher, n: usize) {
    let codes = make_codes(n);
    let threshold: u16 = 512;
    bencher.bench(|| {
        let mut out: Vec<bool> = Vec::with_capacity(n);
        for &c in &codes {
            out.push(c < threshold);
        }
        black_box(out)
    });
}

#[divan::bench(args = SIZES)]
fn cmp_to_bits(bencher: Bencher, n: usize) {
    let codes = make_codes(n);
    let threshold: u16 = 512;
    bencher.bench(|| {
        let bits = BitBuffer::collect_bool(n, |i| {
            // SAFETY: i in 0..n, codes len == n.
            unsafe { *codes.get_unchecked(i) < threshold }
        });
        black_box(bits)
    });
}

// Hand-rolled chunked cmp that the compiler can vectorize cleanly: process 8 codes per
// iteration, pack into a single byte directly.
#[divan::bench(args = SIZES)]
fn cmp_to_bits_chunked(bencher: Bencher, n: usize) {
    let codes = make_codes(n);
    let threshold: u16 = 512;
    bencher.bench(|| {
        let bytes_len = n.div_ceil(8);
        let mut bytes = vec![0u8; bytes_len];
        // Process 8 codes per output byte.
        let full_chunks = n / 8;
        for chunk_idx in 0..full_chunks {
            let base = chunk_idx * 8;
            let mut b = 0u8;
            for j in 0..8 {
                let c = unsafe { *codes.get_unchecked(base + j) };
                b |= u8::from(c < threshold) << j;
            }
            bytes[chunk_idx] = b;
        }
        // Tail
        let tail = full_chunks * 8;
        if tail < n {
            let mut b = 0u8;
            for j in 0..(n - tail) {
                let c = unsafe { *codes.get_unchecked(tail + j) };
                b |= u8::from(c < threshold) << j;
            }
            bytes[full_chunks] = b;
        }
        black_box(bytes)
    });
}
