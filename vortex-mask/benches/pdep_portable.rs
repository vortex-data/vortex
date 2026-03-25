// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark intersect_by_rank implementations across sizes and densities.
//!
//! Variants:
//! - `baseline_portable`: bit-serial PDEP + branchy extract (old develop fallback)
//! - `best_non_bmi2`: LUT PDEP + flat mask sliding window (new portable path, good on Apple M-series)
//! - `production`: current `Mask::intersect_by_rank` (BMI2+SHRD on x86_64, LUT on others)
//! - `production_owned`: `Mask::intersect_by_rank_owned` (in-place when sole owner)

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::identity_op
)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_mask::AllOr;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

fn create_random_mask(len: usize, selectivity: f64) -> Mask {
    Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| {
        let threshold = (selectivity * 1000.0) as usize;
        (i * 7 + 13) % 1000 < threshold
    })))
}

// ── Old baseline (bit-serial PDEP + branchy extract) ─────────────────────────

#[inline]
fn pdep_serial(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    while mask != 0 {
        let lowest_bit = mask & mask.wrapping_neg();
        if source & 1 != 0 {
            result |= lowest_bit;
        }
        source >>= 1;
        mask &= mask - 1;
    }
    result
}

#[inline]
fn extract_bits_branchy(chunks: &[u64], remainder: u64, start: usize) -> u64 {
    let chunk_idx = start / 64;
    let bit_offset = start % 64;
    let num_full_chunks = chunks.len();

    let first_chunk = if chunk_idx < num_full_chunks {
        unsafe { *chunks.get_unchecked(chunk_idx) }
    } else {
        remainder
    };

    if bit_offset == 0 {
        first_chunk
    } else {
        let second_chunk = if chunk_idx + 1 < num_full_chunks {
            unsafe { *chunks.get_unchecked(chunk_idx + 1) }
        } else if chunk_idx + 1 == num_full_chunks {
            remainder
        } else {
            0
        };
        (first_chunk >> bit_offset) | (second_chunk << (64 - bit_offset))
    }
}

fn intersect_baseline(base: &Mask, filter: &Mask) -> Mask {
    let self_buffer = match base.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return filter.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };
    let mask_buffer = match filter.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return base.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };

    let len = base.len();
    let num_chunks = len.div_ceil(64);
    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
    let mut rank = 0usize;

    let self_chunks = self_buffer.chunks();
    let mask_chunks = mask_buffer.chunks();
    let mask_chunk_vec: Vec<u64> = mask_chunks.iter().collect();
    let mask_remainder = mask_chunks.remainder_bits();

    for self_chunk in self_chunks.iter() {
        let popcount = self_chunk.count_ones() as usize;
        let result_chunk = if self_chunk == 0 {
            0u64
        } else if self_chunk == u64::MAX {
            extract_bits_branchy(&mask_chunk_vec, mask_remainder, rank)
        } else {
            let rank_bits = extract_bits_branchy(&mask_chunk_vec, mask_remainder, rank);
            pdep_serial(rank_bits, self_chunk)
        };
        rank += popcount;
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    let remainder = len % 64;
    if remainder != 0 {
        let self_chunk = self_chunks.remainder_bits();
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_branchy(&mask_chunk_vec, mask_remainder, rank);
            pdep_serial(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
}

// ── New portable path (LUT PDEP + flat mask sliding window) ──────────────────

/// 64K LUT for byte-level PDEP.
struct PdepLut {
    table: [[u8; 256]; 256],
    counts: [u8; 256],
}

impl PdepLut {
    const fn new() -> Self {
        let mut table = [[0u8; 256]; 256];
        let mut counts = [0u8; 256];
        let mut mask_byte = 0usize;
        while mask_byte < 256 {
            let mut m = mask_byte as u8;
            let mut c = 0u8;
            while m != 0 {
                c += 1;
                m &= m.wrapping_sub(1);
            }
            counts[mask_byte] = c;
            let mut source_val = 0usize;
            while source_val < 256 {
                let mut src = source_val as u8;
                let mut m = mask_byte as u8;
                let mut res = 0u8;
                while m != 0 {
                    let lowest = m & m.wrapping_neg();
                    if src & 1 != 0 {
                        res |= lowest;
                    }
                    src >>= 1;
                    m &= m.wrapping_sub(1);
                }
                table[mask_byte][source_val] = res;
                source_val += 1;
            }
            mask_byte += 1;
        }
        PdepLut { table, counts }
    }
}

static LUT: PdepLut = PdepLut::new();

#[inline]
fn pdep_lut(mut source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    for byte_idx in 0..8u32 {
        let mask_byte = ((mask >> (byte_idx * 8)) & 0xFF) as usize;
        if mask_byte == 0 {
            continue;
        }
        let count = LUT.counts[mask_byte];
        let src_byte = (source & 0xFF) as usize;
        result |= (LUT.table[mask_byte][src_byte] as u64) << (byte_idx * 8);
        source >>= count;
    }
    result
}

#[inline]
fn extract_bits_flat(mask_flat: &[u64], bit_pos: usize) -> u64 {
    let chunk_idx = bit_pos >> 6;
    let shift = (bit_pos & 63) as u32;
    let lo = unsafe { *mask_flat.get_unchecked(chunk_idx) };
    let hi = unsafe { *mask_flat.get_unchecked(chunk_idx + 1) };
    let mask = (shift != 0) as u64 * u64::MAX;
    (lo >> shift) | ((hi << ((64u32.wrapping_sub(shift)) & 63)) & mask)
}

fn build_mask_flat(buf: &BitBuffer) -> Vec<u64> {
    let chunks = buf.chunks();
    let total = chunks.chunk_len() + 2;
    let mut flat = Vec::with_capacity(total);
    let bytes = buf.inner().as_slice();
    if buf.offset() == 0 && bytes.as_ptr().align_offset(align_of::<u64>()) == 0 {
        let num_full = buf.len() / 64;
        let raw = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u64>(), num_full) };
        flat.extend_from_slice(raw);
        if !buf.len().is_multiple_of(64) {
            let rem_bytes = &bytes[num_full * 8..];
            let mut val = 0u64;
            for (i, &b) in rem_bytes.iter().enumerate() {
                val |= (b as u64) << (i * 8);
            }
            flat.push(val & ((1u64 << (buf.len() % 64)) - 1));
        } else {
            flat.push(0);
        }
    } else {
        flat.extend(chunks.iter());
        flat.push(chunks.remainder_bits());
    }
    flat.push(0); // sentinel
    flat
}

fn intersect_lut_flat(base: &Mask, filter: &Mask) -> Mask {
    let self_buffer = match base.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return filter.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };
    let mask_buffer = match filter.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return base.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };

    let len = base.len();
    let mask_flat = build_mask_flat(mask_buffer);
    let self_chunks = self_buffer.chunks();
    let has_remainder = !len.is_multiple_of(64);
    let num_out = self_chunks.chunk_len() + has_remainder as usize;
    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
    let mut bit_pos = 0usize;

    for self_chunk in self_chunks.iter() {
        let popcount = self_chunk.count_ones() as usize;
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_flat(&mask_flat, bit_pos);
            pdep_lut(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
        bit_pos += popcount;
    }

    if has_remainder {
        let self_chunk = self_chunks.remainder_bits();
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_flat(&mask_flat, bit_pos);
            pdep_lut(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
}

// ── Nibble LUT PDEP (256 bytes, guaranteed L1-hot) ──────────────────────────

/// 256-byte LUT for nibble-level PDEP.
/// 16 iterations per u64 chunk but entire table fits in 4 cache lines.
struct PdepNibbleLut {
    table: [[u8; 16]; 16],
    counts: [u8; 16],
}

impl PdepNibbleLut {
    const fn new() -> Self {
        let mut table = [[0u8; 16]; 16];
        let mut counts = [0u8; 16];
        let mut mask_nib = 0usize;
        while mask_nib < 16 {
            let mut m = mask_nib as u8;
            let mut c = 0u8;
            while m != 0 {
                c += 1;
                m &= m.wrapping_sub(1);
            }
            counts[mask_nib] = c;
            let mut src_val = 0usize;
            while src_val < 16 {
                let mut src = src_val as u8;
                let mut m = mask_nib as u8;
                let mut res = 0u8;
                while m != 0 {
                    let lowest = m & m.wrapping_neg();
                    if src & 1 != 0 {
                        res |= lowest;
                    }
                    src >>= 1;
                    m &= m.wrapping_sub(1);
                }
                table[mask_nib][src_val] = res;
                src_val += 1;
            }
            mask_nib += 1;
        }
        PdepNibbleLut { table, counts }
    }
}

static NIBBLE_LUT: PdepNibbleLut = PdepNibbleLut::new();

#[inline]
fn pdep_nibble(mut source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    for nib_idx in 0..16u32 {
        let mask_nib = ((mask >> (nib_idx * 4)) & 0xF) as usize;
        if mask_nib == 0 {
            continue;
        }
        let count = NIBBLE_LUT.counts[mask_nib];
        let src_nib = (source & 0xF) as usize;
        result |= (NIBBLE_LUT.table[mask_nib][src_nib] as u64) << (nib_idx * 4);
        source >>= count;
    }
    result
}

fn intersect_nibble_flat(base: &Mask, filter: &Mask) -> Mask {
    let self_buffer = match base.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return filter.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };
    let mask_buffer = match filter.bit_buffer() {
        AllOr::Some(b) => b,
        AllOr::All => return base.clone(),
        AllOr::None => return Mask::new_false(base.len()),
    };

    let len = base.len();
    let mask_flat = build_mask_flat(mask_buffer);
    let self_chunks = self_buffer.chunks();
    let has_remainder = !len.is_multiple_of(64);
    let num_out = self_chunks.chunk_len() + has_remainder as usize;
    let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_out);
    let mut bit_pos = 0usize;

    for self_chunk in self_chunks.iter() {
        let popcount = self_chunk.count_ones() as usize;
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_flat(&mask_flat, bit_pos);
            pdep_nibble(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
        bit_pos += popcount;
    }

    if has_remainder {
        let self_chunk = self_chunks.remainder_bits();
        let result_chunk = if self_chunk == 0 {
            0u64
        } else {
            let rank_bits = extract_bits_flat(&mask_flat, bit_pos);
            pdep_nibble(rank_bits, self_chunk)
        };
        unsafe { buffer.push_unchecked(result_chunk) };
    }

    buffer.truncate(len.div_ceil(8));
    Mask::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
}

// ── Production owned (in-place when possible) ───────────────────────────────

// ── Benchmark parameters ─────────────────────────────────────────────────────

const BENCH_CASES: &[(usize, f64, &str)] = &[
    (100_000, 0.10, "100K_10pct"),
    (100_000, 0.50, "100K_50pct"),
    (100_000, 0.90, "100K_90pct"),
    (1_000_000, 0.10, "1M_10pct"),
    (1_000_000, 0.50, "1M_50pct"),
    (1_000_000, 0.90, "1M_90pct"),
    (10_000_000, 0.10, "10M_10pct"),
    (10_000_000, 0.50, "10M_50pct"),
    (10_000_000, 0.90, "10M_90pct"),
    (100_000_000, 0.50, "100M_50pct"),
];

// ── Benchmarks ───────────────────────────────────────────────────────────────

#[divan::bench(args = BENCH_CASES)]
fn baseline_portable(bencher: Bencher, &(size, density, _): &(usize, f64, &str)) {
    let base = create_random_mask(size, density);
    let rank = create_random_mask(base.true_count(), 0.5);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(b, r)| intersect_baseline(b, r));
}

#[divan::bench(args = BENCH_CASES)]
fn best_non_bmi2(bencher: Bencher, &(size, density, _): &(usize, f64, &str)) {
    let base = create_random_mask(size, density);
    let rank = create_random_mask(base.true_count(), 0.5);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(b, r)| intersect_lut_flat(b, r));
}

#[divan::bench(args = BENCH_CASES)]
fn nibble_lut(bencher: Bencher, &(size, density, _): &(usize, f64, &str)) {
    let base = create_random_mask(size, density);
    let rank = create_random_mask(base.true_count(), 0.5);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(b, r)| intersect_nibble_flat(b, r));
}

#[divan::bench(args = BENCH_CASES)]
fn production(bencher: Bencher, &(size, density, _): &(usize, f64, &str)) {
    let base = create_random_mask(size, density);
    let rank = create_random_mask(base.true_count(), 0.5);
    bencher
        .with_inputs(|| (&base, &rank))
        .bench_refs(|(b, r)| b.intersect_by_rank(r));
}

#[divan::bench(args = BENCH_CASES)]
fn production_owned(bencher: Bencher, &(size, density, _): &(usize, f64, &str)) {
    let base = create_random_mask(size, density);
    let rank = create_random_mask(base.true_count(), 0.5);
    // Create a unique BitBuffer so try_into_mut succeeds (no shared Arc).
    let base_buf = base.to_bit_buffer();
    bencher
        .with_inputs(|| (Mask::from_buffer(base_buf.clone()), &rank))
        .bench_values(|(b, r)| b.intersect_by_rank_owned(r));
}
