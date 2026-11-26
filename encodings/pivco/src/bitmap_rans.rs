// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-node skewed-bitmap coder (`PHA` layer for `vortex-pivco`).
//!
//! Each internal wavelet-tree node of `n` elements produces a `ceil(n/8)`-byte partition bitmap.
//! Near-balanced bitmaps are emitted raw (`0` bytes header beyond 1 marker byte, decoded straight
//! from the wire buffer with zero copy). Skewed bitmaps compress via:
//! 1. **Constant (`AllZero` / `AllOne`)**: `0` payload bytes.
//! 2. **Sparse minority bit indices (`SparseOne` / `SparseZero`)**: `2` bytes per minority bit when
//!    density `<= 1/64`, decoded via `memset` + bit scatter (or scanned directly in `O(count)` for
//!    top-down `scalar_at`).
//! 3. **Pre-built static Bernoulli byte-rANS (`StaticByteRans`)**: 16 static byte-alphabet tables
//!    built with exact integer arithmetic at rational minority bit probabilities `p_k = (k + 1) / 64`
//!    for `k in 0..16` (`1` recipe byte on wire -> zero table header and bit-exact across
//!    languages/architectures), decoded with canonical `32`-lane (`1` CUDA warp) interleaved rANS
//!    at **8 bits per rANS step**.

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;
use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::simd::popcount_bits;

/// Marker values for serialized node bitmaps.
pub(crate) const BM_MARKER_RAW: u8 = 0;
pub(crate) const BM_MARKER_ALL_ZERO: u8 = 1;
pub(crate) const BM_MARKER_ALL_ONE: u8 = 2;
pub(crate) const BM_MARKER_SPARSE_ONE: u8 = 3;
pub(crate) const BM_MARKER_SPARSE_ZERO: u8 = 4;
/// `0x80 | (xor_flip << 6) | table_id` (`table_id in 0..16`; reserved bits `0x30` must be `0`).
pub(crate) const BM_MARKER_RANS_BIT: u8 = 0x80;
/// Reserved marker bits inside a rANS marker byte that must always be zero.
const BM_MARKER_RANS_RESERVED_MASK: u8 = 0x30;

/// Canonical 32-lane interleaved byte-rANS width (`1` CUDA warp).
pub(crate) const BM_LANES: usize = 32;
/// Byte size of the 32 initial lane states (`32 * 4 = 128` bytes).
const BM_RANS_STATE_BYTES: usize = BM_LANES * 4;

/// Shared PHA bitmap-mode gating constants used by both encoder and size estimator.
const BM_PHA_MIN_BYTES: usize = 16;
const BM_SPARSE_MAX_RATIO_INV: usize = 64;
const BM_SPARSE_MIN_SAVINGS_BYTES: usize = 8;
const BM_RANS_MIN_BYTES: usize = BM_RANS_STATE_BYTES + 16;
const BM_RANS_MAX_MINOR_PCT: usize = 34;
const BM_RANS_MIN_SAVINGS_BYTES: usize = 2;
const BM_RANS_SPEED_GATE_PCT: usize = 94;

const BM_PROB_BITS: u32 = 11;
const BM_PROB_SCALE: u32 = 1 << BM_PROB_BITS;
const BM_PROB_MASK: u32 = BM_PROB_SCALE - 1;
const BM_RANS_L: u32 = 1 << 15;
const BM_RECIP_SHIFT: u32 = 43;
const NUM_STATIC_TABLES: usize = 16;

/// Denominator `64` for exact rational minority-bit probabilities `p_k = (k + 1) / 64`, `k in 0..16`.
const BERNOULLI_DENOM: u64 = 64;
/// Exact sum of all 256 8-bit Bernoulli weights `(p_num + (64 - p_num))^8 = 64^8 = 2^48`.
const BERNOULLI_WEIGHT_SUM: u128 = 1u128 << 48;

#[derive(Clone, Copy, Default)]
struct ByteDecodeSlot {
    freq: u16,
    bias: u16,
    byte_sym: u8,
    _pad: [u8; 3],
}

#[derive(Clone, Copy, Default)]
struct ByteEncodeInfo {
    freq: u32,
    cumfreq: u32,
    x_max: u32,
    recip: u64,
}

struct StaticByteTable {
    enc: Box<[ByteEncodeInfo; 256]>,
    dec: Box<[ByteDecodeSlot; BM_PROB_SCALE as usize]>,
}

static STATIC_TABLES: LazyLock<Vec<StaticByteTable>> = LazyLock::new(|| {
    (0..NUM_STATIC_TABLES)
        .map(|table_idx| build_bernoulli_table((table_idx + 1) as u64))
        .collect()
});

/// Computes the exact quantized 256-entry frequency table (`sum == 2048`, each `>= 1`) for rational
/// minority bit probability `p = p_num / 64` using pure integer arithmetic (zero `f64`).
pub(crate) fn compute_bernoulli_freqs(p_num: u64) -> [u16; 256] {
    debug_assert!((1..BERNOULLI_DENOM).contains(&p_num));
    let p_zero = BERNOULLI_DENOM - p_num;

    let mut freqs = [1u16; 256];
    let mut sum_f = 0i32;
    let mut order: [(usize, i128); 256] = [(0, 0); 256];

    for byte_val in 0..256usize {
        let ones = (byte_val as u8).count_ones();
        let zeros = 8 - ones;
        let weight = p_num.pow(ones) * p_zero.pow(zeros);
        let scaled_num = u128::from(weight) * u128::from(BM_PROB_SCALE);
        let floor_q = (scaled_num / BERNOULLI_WEIGHT_SUM) as u16;
        let quantized = floor_q.max(1);
        freqs[byte_val] = quantized;
        sum_f += i32::from(quantized);
        let residual =
            (scaled_num as i128) - (i128::from(quantized) * (BERNOULLI_WEIGHT_SUM as i128));
        order[byte_val] = (byte_val, residual);
    }

    let target = BM_PROB_SCALE as i32;
    if sum_f < target {
        order.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        for &(sym, _) in order.iter().take((target - sum_f) as usize) {
            freqs[sym] += 1;
        }
    } else if sum_f > target {
        order.sort_by(|lhs, rhs| {
            lhs.1
                .cmp(&rhs.1)
                .then_with(|| freqs[rhs.0].cmp(&freqs[lhs.0]))
                .then_with(|| lhs.0.cmp(&rhs.0))
        });
        let mut surplus = (sum_f - target) as usize;
        while surplus > 0 {
            for &(sym, _) in &order {
                if surplus == 0 {
                    break;
                }
                if freqs[sym] > 1 {
                    freqs[sym] -= 1;
                    surplus -= 1;
                }
            }
        }
    }

    freqs
}

fn build_bernoulli_table(p_num: u64) -> StaticByteTable {
    let freqs = compute_bernoulli_freqs(p_num);

    let mut enc: Box<[ByteEncodeInfo; 256]> = vec![ByteEncodeInfo::default(); 256]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!());
    let mut dec: Box<[ByteDecodeSlot; BM_PROB_SCALE as usize]> =
        vec![ByteDecodeSlot::default(); BM_PROB_SCALE as usize]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!());
    let mut cum = 0u32;
    for (byte_val, &freq) in freqs.iter().enumerate() {
        let freq_u32 = u32::from(freq);
        let recip = if freq_u32 == 1 {
            1u64 << BM_RECIP_SHIFT
        } else {
            ((1u64 << BM_RECIP_SHIFT) / u64::from(freq_u32)) + 1
        };
        enc[byte_val] = ByteEncodeInfo {
            freq: freq_u32,
            cumfreq: cum,
            x_max: freq_u32 << 20,
            recip,
        };
        for rem in 0..freq_u32 {
            dec[(cum + rem) as usize] = ByteDecodeSlot {
                freq,
                bias: rem as u16,
                byte_sym: byte_val as u8,
                _pad: [0; 3],
            };
        }
        cum += freq_u32;
    }
    debug_assert_eq!(cum, BM_PROB_SCALE);

    StaticByteTable { enc, dec }
}

/// Shared sparse-mode eligibility predicate used by both encoder and cost estimator.
#[inline]
fn sparse_mode_eligible(
    len: usize,
    n_minor: usize,
    sparse_after_marker: usize,
    nbytes: usize,
) -> bool {
    len <= usize::from(u16::MAX)
        && sparse_after_marker + BM_SPARSE_MIN_SAVINGS_BYTES < nbytes
        && n_minor
            .checked_mul(BM_SPARSE_MAX_RATIO_INV)
            .is_some_and(|v| v <= len)
}

/// Shared rANS pre-check used by both encoder and cost estimator.
#[inline]
fn rans_mode_eligible(nbytes: usize, len: usize, n_minor: usize) -> bool {
    nbytes >= BM_RANS_MIN_BYTES && n_minor * 100 <= len * BM_RANS_MAX_MINOR_PCT
}

/// Shared speed-aware commit gate for 32-lane static byte-rANS used by both encoder and estimator.
#[inline]
fn rans_passes_commit_gate(
    len: usize,
    depth: usize,
    coded_len: usize,
    total_after_marker: usize,
    nbytes: usize,
) -> bool {
    if coded_len > usize::from(u16::MAX) || total_after_marker + BM_RANS_MIN_SAVINGS_BYTES >= nbytes
    {
        return false;
    }
    let lhs = 100u128 * (depth as u128 * len as u128 + total_after_marker as u128 * 8);
    let rhs = (BM_RANS_SPEED_GATE_PCT as u128) * ((depth + 1) as u128) * (len as u128);
    lhs <= rhs
}

/// Estimates compressed bit length (including the 1-byte marker and alignment padding) for one
/// internal wavelet-tree node bitmap of `len` elements and `k_right` right-going elements using
/// the exact shared `PHA` gating constants.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) fn estimate_node_bitmap_bits(
    len: usize,
    k_right: usize,
    depth: usize,
    pha_enabled: bool,
) -> usize {
    if len == 0 {
        return 0;
    }
    if k_right == 0 || k_right == len {
        return 8;
    }

    let nbytes = len.div_ceil(8);
    let raw_bits = (1 + nbytes) * 8;
    if !pha_enabled || nbytes < BM_PHA_MIN_BYTES {
        return raw_bits;
    }

    let n_minor = k_right.min(len - k_right);
    // Expected alignment pad for sparse header is 0..=1 byte (conservatively use 1 byte).
    let sparse_after_marker = 1 + 2 + n_minor * 2;
    if sparse_mode_eligible(len, n_minor, sparse_after_marker, nbytes) {
        return (1 + sparse_after_marker) * 8;
    }

    if rans_mode_eligible(nbytes, len, n_minor) {
        let p = (n_minor as f64) / (len as f64);
        let h2 = if p > 0.0 && p < 1.0 {
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        } else {
            0.0
        };
        let entropy_bytes = ((len as f64) * h2 / 8.0).ceil() as usize;
        let coded_len = BM_RANS_STATE_BYTES + entropy_bytes;
        // Expected alignment pad for 4-byte aligned states is 0..=3 bytes (conservatively use 2).
        let total_after_marker = 2 + 2 + coded_len;
        if rans_passes_commit_gate(len, depth, coded_len, total_after_marker, nbytes) {
            return (1 + total_after_marker) * 8;
        }
    }

    raw_bits
}

/// Emits strictly increasing `u16` minority bit positions `< len` via word-wise `trailing_zeros`.
fn write_sparse_indices(bm: &[u8], len: usize, xor_flip: bool, dst: &mut Vec<u8>) {
    let nbytes = len.div_ceil(8);
    let (u64_chunks, tail) = bm[..nbytes].as_chunks::<8>();
    let mut base_bit = 0usize;

    for &chunk in u64_chunks {
        let raw = u64::from_le_bytes(chunk);
        let mut word = if xor_flip { !raw } else { raw };
        let remaining = len - base_bit;
        if remaining < 64 {
            word &= (1u64 << remaining) - 1;
        }
        while word != 0 {
            let tz = word.trailing_zeros() as usize;
            dst.extend_from_slice(&((base_bit + tz) as u16).to_le_bytes());
            word &= word - 1;
        }
        base_bit += 64;
    }

    for &byte in tail {
        let mut b = if xor_flip { !byte } else { byte };
        let remaining = len - base_bit;
        if remaining < 8 {
            b &= (1u8 << remaining) - 1;
        }
        while b != 0 {
            let tz = b.trailing_zeros() as usize;
            dst.extend_from_slice(&((base_bit + tz) as u16).to_le_bytes());
            b &= b - 1;
        }
        base_bit += 8;
    }
}

/// Writes an `n`-bit node bitmap into `dst` using canonical 32-lane static byte-rANS when `PHA`
/// is enabled and profitable, reusing `rans_words` to avoid per-node heap allocations.
pub(crate) fn write_node_bitmap(
    bm: &[u8],
    len: usize,
    k_right: usize,
    depth: usize,
    pha_enabled: bool,
    rans_words: &mut Vec<u16>,
    dst: &mut Vec<u8>,
) {
    let nbytes = len.div_ceil(8);
    if k_right == 0 {
        dst.push(BM_MARKER_ALL_ZERO);
        return;
    }
    if k_right == len {
        dst.push(BM_MARKER_ALL_ONE);
        return;
    }

    if !pha_enabled || nbytes < BM_PHA_MIN_BYTES {
        dst.push(BM_MARKER_RAW);
        dst.extend_from_slice(&bm[..nbytes]);
        return;
    }

    let k_left = len - k_right;
    let (n_minor, xor_flip) = if k_right <= k_left {
        (k_right, false)
    } else {
        (k_left, true)
    };

    let marker_pos = dst.len();

    // 1. Ultra-sparse minority bit list (`<= 1.56%` minority): pad 0..=1 byte so `n_minor: u16`
    //    and `indices: [u16]` start at an even (`% 2 == 0`) offset for natural `u16` loads.
    let sparse_pad = (marker_pos + 1) & 1;
    let sparse_after_marker = sparse_pad + 2 + n_minor * 2;
    if sparse_mode_eligible(len, n_minor, sparse_after_marker, nbytes) {
        dst.push(if xor_flip {
            BM_MARKER_SPARSE_ZERO
        } else {
            BM_MARKER_SPARSE_ONE
        });
        if sparse_pad > 0 {
            dst.push(0);
        }
        debug_assert!(dst.len().is_multiple_of(2));
        dst.extend_from_slice(&(n_minor as u16).to_le_bytes());
        write_sparse_indices(bm, len, xor_flip, dst);
        return;
    }

    // 2. Canonical 32-lane static Bernoulli byte-rANS (`1` CUDA warp): pad 0..=3 bytes so
    //    `coded_len: u16` sits at `states_pos - 2` (`% 2 == 0`) and `states: [u32; 32]` +
    //    `words: [u16]` start at `states_pos` (`% 4 == 0`) for natural `u32`/`u16` loads.
    if rans_mode_eligible(nbytes, len, n_minor) {
        let p64_rounded = (n_minor * (BERNOULLI_DENOM as usize * 2) + len) / (2 * len);
        let table_idx = p64_rounded.saturating_sub(1).min(NUM_STATIC_TABLES - 1);
        let states =
            encode_static_byte_rans_into(&bm[..nbytes], len, xor_flip, table_idx, rans_words);
        let coded_len = BM_RANS_STATE_BYTES + rans_words.len() * 2;
        let states_pos = (marker_pos + 6) & !3;
        let coded_len_pos = states_pos - 2;
        let rans_pad = coded_len_pos - (marker_pos + 1);
        let total_after_marker = rans_pad + 2 + coded_len;
        if rans_passes_commit_gate(len, depth, coded_len, total_after_marker, nbytes) {
            let marker = BM_MARKER_RANS_BIT | (u8::from(xor_flip) << 6) | (table_idx as u8);
            dst.push(marker);
            if rans_pad > 0 {
                dst.extend_from_slice(&[0u8; 3][..rans_pad]);
            }
            debug_assert_eq!(dst.len(), coded_len_pos);
            debug_assert!(dst.len().is_multiple_of(2));
            dst.extend_from_slice(&(coded_len as u16).to_le_bytes());
            debug_assert_eq!(dst.len(), states_pos);
            debug_assert!(dst.len().is_multiple_of(4));
            dst.reserve(coded_len);
            for state in states {
                dst.extend_from_slice(&state.to_le_bytes());
            }
            for &word in rans_words.iter().rev() {
                dst.extend_from_slice(&word.to_le_bytes());
            }
            return;
        }
    }

    dst.push(BM_MARKER_RAW);
    dst.extend_from_slice(&bm[..nbytes]);
}

/// Encodes `bm[..ceil(len/8)]` using canonical 32-lane interleaved static Bernoulli byte-rANS,
/// writing emitted renormalization words in reverse into `emitted_rev` and returning final lane
/// states.
pub(crate) fn encode_static_byte_rans_into(
    bm: &[u8],
    len: usize,
    xor_flip: bool,
    table_idx: usize,
    emitted_rev: &mut Vec<u16>,
) -> [u32; BM_LANES] {
    emitted_rev.clear();
    let table = &STATIC_TABLES[table_idx];
    let nbytes = bm.len();
    let tail_mask = if (len & 7) == 0 {
        0xFFu8
    } else {
        (1u8 << (len & 7)) - 1
    };

    let n_stripes = nbytes.div_ceil(BM_LANES);
    let mut states = [BM_RANS_L; BM_LANES];

    for stripe in (0..n_stripes).rev() {
        let base = stripe * BM_LANES;
        for lane in (0..BM_LANES).rev() {
            let pos = base + lane;
            let sym_byte = if pos < nbytes {
                let mut raw = if xor_flip { !bm[pos] } else { bm[pos] };
                if pos + 1 == nbytes {
                    raw &= tail_mask;
                }
                raw
            } else {
                0u8
            };
            let info = table.enc[usize::from(sym_byte)];
            let mut state = states[lane];
            if state >= info.x_max {
                emitted_rev.push(state as u16);
                state >>= 16;
            }
            let quot = ((u64::from(state).wrapping_mul(info.recip)) >> BM_RECIP_SHIFT) as u32;
            let rem = state - quot * info.freq;
            states[lane] = (quot << BM_PROB_BITS) + info.cumfreq + rem;
        }
    }

    states
}

/// Decodes a canonical 32-lane interleaved static Bernoulli byte-rANS stream into `out[..nbytes]`,
/// verifying stream exhaustion (`word_pos == words.len()`) and terminal lane states (`== BM_RANS_L`).
pub(crate) fn decode_static_byte_rans<'o>(
    payload: &[u8],
    len: usize,
    xor_flip: bool,
    table_idx: usize,
    out: &'o mut [MaybeUninit<u8>],
) -> VortexResult<&'o mut [u8]> {
    vortex_ensure!(
        payload.len() >= BM_RANS_STATE_BYTES
            && (payload.len() - BM_RANS_STATE_BYTES).is_multiple_of(2),
        "Invalid PIVCO byte-rANS payload length {}",
        payload.len()
    );
    let table = &STATIC_TABLES[table_idx];
    let dec = &table.dec;
    let mut states = [0u32; BM_LANES];
    for (lane_state, chunk) in states
        .iter_mut()
        .zip(payload[..BM_RANS_STATE_BYTES].as_chunks::<4>().0.iter())
    {
        let s = u32::from_le_bytes(*chunk);
        vortex_ensure!(s >= BM_RANS_L, "Invalid PIVCO byte-rANS initial state {s}");
        *lane_state = s;
    }
    let words = payload[BM_RANS_STATE_BYTES..].as_chunks::<2>().0;
    let mut word_pos = 0usize;
    let nbytes = out.len();
    let n_stripes = nbytes.div_ceil(BM_LANES);
    let xor_mask = if xor_flip { 0xFFu8 } else { 0x00u8 };

    for stripe in 0..n_stripes {
        let stripe_base = stripe * BM_LANES;
        let mut sub_off = 0usize;
        while sub_off < BM_LANES {
            let base = stripe_base + sub_off;
            let mut next = [0u32; 8];

            if base + 8 <= nbytes {
                let dst = &mut out[base..base + 8];
                for lane in 0..8 {
                    let state = states[sub_off + lane];
                    let entry = dec[(state & BM_PROB_MASK) as usize];
                    dst[lane].write(entry.byte_sym ^ xor_mask);
                    next[lane] =
                        u32::from(entry.freq) * (state >> BM_PROB_BITS) + u32::from(entry.bias);
                }
            } else {
                let mut syms = [0u8; 8];
                for lane in 0..8 {
                    let state = states[sub_off + lane];
                    let entry = dec[(state & BM_PROB_MASK) as usize];
                    syms[lane] = entry.byte_sym ^ xor_mask;
                    next[lane] =
                        u32::from(entry.freq) * (state >> BM_PROB_BITS) + u32::from(entry.bias);
                }
                if base < nbytes {
                    let valid = (nbytes - base).min(8);
                    for j in 0..valid {
                        out[base + j].write(syms[j]);
                    }
                }
            }

            if word_pos + 8 <= words.len() {
                for lane in 0..8 {
                    let state = next[lane];
                    let need = state < BM_RANS_L;
                    // SAFETY: `word_pos + 8 <= words.len()` and `word_pos` increments at most 8 times.
                    let word = u32::from(u16::from_le_bytes(*unsafe {
                        words.get_unchecked(word_pos)
                    }));
                    states[sub_off + lane] = if need { (state << 16) | word } else { state };
                    word_pos += usize::from(need);
                }
            } else {
                for lane in 0..8 {
                    let mut state = next[lane];
                    if state < BM_RANS_L {
                        vortex_ensure!(
                            word_pos < words.len(),
                            "Truncated PIVCO byte-rANS renormalization word stream"
                        );
                        let word = u32::from(u16::from_le_bytes(words[word_pos]));
                        word_pos += 1;
                        state = (state << 16) | word;
                    }
                    states[sub_off + lane] = state;
                }
            }

            sub_off += 8;
        }
    }

    vortex_ensure!(
        word_pos == words.len(),
        "Trailing words in PIVCO byte-rANS stream ({word_pos} != {})",
        words.len()
    );
    vortex_ensure!(
        states.iter().all(|&s| s == BM_RANS_L),
        "Corrupt PIVCO byte-rANS terminal lane state"
    );

    // SAFETY: Every element in `0..nbytes` was written above across `0..n_stripes`.
    let init = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), nbytes) };
    if (len & 7) != 0 && nbytes > 0 {
        init[nbytes - 1] &= (1u8 << (len & 7)) - 1;
    }
    Ok(init)
}

/// Validated sparse minority-index list (`count <= len`, `count * 64 <= len`, strictly increasing
/// indices `< len`).
fn parse_sparse<'a>(wire: &'a [u8], cursor: &mut usize, len: usize) -> VortexResult<&'a [u8]> {
    let pad = *cursor & 1;
    vortex_ensure!(
        *cursor + pad + 2 <= wire.len(),
        "Truncated PIVCO sparse count"
    );
    *cursor += pad;
    let count = usize::from(u16::from_le_bytes([wire[*cursor], wire[*cursor + 1]]));
    *cursor += 2;
    vortex_ensure!(
        count <= len
            && count
                .checked_mul(BM_SPARSE_MAX_RATIO_INV)
                .is_some_and(|v| v <= len),
        "Invalid PIVCO sparse index count {count} for length {len}"
    );
    let byte_len = count * 2;
    vortex_ensure!(
        *cursor + byte_len <= wire.len(),
        "Truncated PIVCO sparse indices"
    );
    let idx_bytes = &wire[*cursor..*cursor + byte_len];
    *cursor += byte_len;

    let mut prev: Option<usize> = None;
    for &[lo, hi] in idx_bytes.as_chunks::<2>().0 {
        let idx = usize::from(u16::from_le_bytes([lo, hi]));
        vortex_ensure!(
            idx < len,
            "PIVCO sparse index {idx} out of bounds for length {len}"
        );
        if let Some(p) = prev {
            vortex_ensure!(
                idx > p,
                "PIVCO sparse indices must be strictly increasing ({idx} <= {p})"
            );
        }
        prev = Some(idx);
    }
    Ok(idx_bytes)
}

fn decode_rans_marker_into<'s>(
    marker: u8,
    wire: &[u8],
    cursor: &mut usize,
    len: usize,
    nbytes: usize,
    scratch: &'s mut [MaybeUninit<u8>],
) -> VortexResult<&'s [u8]> {
    vortex_ensure!(
        (marker & BM_MARKER_RANS_RESERVED_MASK) == 0,
        "Invalid PIVCO byte-rANS marker reserved bits {marker:#x}"
    );
    let xor_flip = ((marker >> 6) & 1) != 0;
    let table_idx = usize::from(marker & 0x0F);
    let marker_pos = *cursor - 1;
    let states_pos = (marker_pos + 6) & !3;
    let coded_len_pos = states_pos - 2;
    vortex_ensure!(states_pos <= wire.len(), "Truncated PIVCO byte-rANS header");
    *cursor = coded_len_pos;
    let c_len = usize::from(u16::from_le_bytes([wire[*cursor], wire[*cursor + 1]]));
    *cursor += 2;
    debug_assert_eq!(*cursor, states_pos);
    vortex_ensure!(
        c_len >= BM_RANS_STATE_BYTES
            && (c_len - BM_RANS_STATE_BYTES).is_multiple_of(2)
            && *cursor + c_len <= wire.len(),
        "Invalid PIVCO byte-rANS payload length {c_len}"
    );
    let payload = &wire[*cursor..*cursor + c_len];
    *cursor += c_len;
    let init = decode_static_byte_rans(payload, len, xor_flip, table_idx, &mut scratch[..nbytes])?;
    Ok(init)
}

/// Unified validated node-bitmap representation shared by full block decode, top-down `scalar_at`,
/// prefix `ebits` summation, and spine predicate pushdown.
///
/// Every variant enforces the same wire invariants (`parse_sparse` strict monotonicity/bounds,
/// rANS reserved bits & terminal state checks), so all decode paths agree unconditionally.
pub(crate) enum NodeBitmap<'a> {
    AllZero,
    AllOne,
    Raw(&'a [u8]),
    /// Validated strictly increasing `u16` little-endian minority indices `< len`
    /// (`invert == false` => `SparseOne`, `true` => `SparseZero`).
    Sparse {
        idx_bytes: &'a [u8],
        invert: bool,
    },
    /// Decoded into the first `ceil(len/8)` bytes of the caller's scratch buffer (`StaticByteRans`).
    Scratch,
}

/// Dense byte-slice view of a [`NodeBitmap`] after optional scatter of `Sparse` into `scratch`.
pub(crate) enum DenseBitmap<'a> {
    AllZero,
    AllOne,
    Bytes(&'a [u8]),
}

impl<'a> NodeBitmap<'a> {
    /// Parses one serialized node bitmap of `len` bits from `wire[*cursor..]`, advancing `*cursor`.
    ///
    /// `scratch` is ONLY written (without pre-zeroing) when a `StaticByteRans` marker is encountered.
    pub(crate) fn read(
        wire: &'a [u8],
        cursor: &mut usize,
        len: usize,
        scratch: &mut [MaybeUninit<u8>],
    ) -> VortexResult<Self> {
        vortex_ensure!(*cursor < wire.len(), "Truncated PIVCO node bitmap marker");
        let marker = wire[*cursor];
        *cursor += 1;
        let nbytes = len.div_ceil(8);

        match marker {
            BM_MARKER_ALL_ZERO => Ok(Self::AllZero),
            BM_MARKER_ALL_ONE => Ok(Self::AllOne),
            BM_MARKER_RAW => {
                vortex_ensure!(
                    *cursor + nbytes <= wire.len(),
                    "Truncated PIVCO raw bitmap ({nbytes} bytes)"
                );
                let slice = &wire[*cursor..*cursor + nbytes];
                *cursor += nbytes;
                Ok(Self::Raw(slice))
            }
            BM_MARKER_SPARSE_ONE | BM_MARKER_SPARSE_ZERO => {
                let idx_bytes = parse_sparse(wire, cursor, len)?;
                Ok(Self::Sparse {
                    idx_bytes,
                    invert: marker == BM_MARKER_SPARSE_ZERO,
                })
            }
            code if (code & BM_MARKER_RANS_BIT) != 0 => {
                vortex_ensure!(
                    scratch.len() >= nbytes,
                    "PIVCO bitmap scratch too small ({} < {nbytes})",
                    scratch.len()
                );
                decode_rans_marker_into(code, wire, cursor, len, nbytes, scratch)?;
                Ok(Self::Scratch)
            }
            other => vortex_bail!("Unknown PIVCO bitmap marker {other:#x}"),
        }
    }

    /// Returns the bit at `pos` (`< len`), the count of 1-bits in `0..pos` (`rank1(pos)`), and
    /// the total count of 1-bits in `0..len` (`count1()`).
    ///
    /// `scratch` must be the same buffer passed to [`Self::read`] (only read for `Self::Scratch`).
    #[inline]
    pub(crate) fn bit_and_ranks(
        &self,
        scratch: &[MaybeUninit<u8>],
        len: usize,
        pos: usize,
    ) -> (bool, usize, usize) {
        debug_assert!(pos < len);
        match *self {
            Self::AllZero => (false, 0, 0),
            Self::AllOne => (true, pos, len),
            Self::Raw(bm) => {
                let bit = ((bm[pos >> 3] >> (pos & 7)) & 1) != 0;
                (bit, popcount_bits(bm, pos), popcount_bits(bm, len))
            }
            Self::Sparse { idx_bytes, invert } => {
                let minor_total = idx_bytes.len() / 2;
                let mut minor_before = 0usize;
                let mut is_minor = false;
                for &[lo, hi] in idx_bytes.as_chunks::<2>().0 {
                    let idx = usize::from(u16::from_le_bytes([lo, hi]));
                    if idx < pos {
                        minor_before += 1;
                    } else if idx == pos {
                        is_minor = true;
                        break;
                    } else {
                        break;
                    }
                }
                let bit = is_minor ^ invert;
                let ones_before = if invert {
                    pos - minor_before
                } else {
                    minor_before
                };
                let total_ones = if invert {
                    len - minor_total
                } else {
                    minor_total
                };
                (bit, ones_before, total_ones)
            }
            Self::Scratch => {
                let bm = scratch_as_init(scratch, len.div_ceil(8));
                let bit = ((bm[pos >> 3] >> (pos & 7)) & 1) != 0;
                (bit, popcount_bits(bm, pos), popcount_bits(bm, len))
            }
        }
    }

    /// Returns `(rank1(prefix_len), count1(len))` where `prefix_len <= len`.
    #[inline]
    pub(crate) fn rank1_and_count1(
        &self,
        scratch: &[MaybeUninit<u8>],
        len: usize,
        prefix_len: usize,
    ) -> (usize, usize) {
        debug_assert!(prefix_len <= len);
        match *self {
            Self::AllZero => (0, 0),
            Self::AllOne => (prefix_len, len),
            Self::Raw(bm) => (popcount_bits(bm, prefix_len), popcount_bits(bm, len)),
            Self::Sparse { idx_bytes, invert } => {
                let minor_total = idx_bytes.len() / 2;
                let mut minor_before = 0usize;
                for &[lo, hi] in idx_bytes.as_chunks::<2>().0 {
                    let idx = usize::from(u16::from_le_bytes([lo, hi]));
                    if idx < prefix_len {
                        minor_before += 1;
                    } else {
                        break;
                    }
                }
                if invert {
                    (prefix_len - minor_before, len - minor_total)
                } else {
                    (minor_before, minor_total)
                }
            }
            Self::Scratch => {
                let bm = scratch_as_init(scratch, len.div_ceil(8));
                (popcount_bits(bm, prefix_len), popcount_bits(bm, len))
            }
        }
    }

    /// Returns the bit at `pos` (`< len`).
    #[cfg(test)]
    #[inline]
    pub(crate) fn bit(&self, scratch: &[MaybeUninit<u8>], len: usize, pos: usize) -> bool {
        self.bit_and_ranks(scratch, len, pos).0
    }

    /// Returns the total count of 1-bits in `0..len`.
    #[cfg(test)]
    #[inline]
    pub(crate) fn count1(&self, scratch: &[MaybeUninit<u8>], len: usize) -> usize {
        match *self {
            Self::AllZero => 0,
            Self::AllOne => len,
            Self::Raw(bm) => popcount_bits(bm, len),
            Self::Sparse { idx_bytes, invert } => {
                let count = idx_bytes.len() / 2;
                if invert { len - count } else { count }
            }
            Self::Scratch => popcount_bits(scratch_as_init(scratch, len.div_ceil(8)), len),
        }
    }

    /// Materializes `Sparse` bitmaps into `scratch[..ceil(len/8)]` for dense SIMD merge kernels,
    /// returning a zero-copy slice for `Raw` and the initialized scratch slice for `Sparse`/`Scratch`.
    #[inline]
    pub(crate) fn into_dense<'s>(
        self,
        scratch: &'s mut [MaybeUninit<u8>],
        len: usize,
    ) -> DenseBitmap<'s>
    where
        'a: 's,
    {
        let nbytes = len.div_ceil(8);
        match self {
            Self::AllZero => DenseBitmap::AllZero,
            Self::AllOne => DenseBitmap::AllOne,
            Self::Raw(slice) => DenseBitmap::Bytes(slice),
            Self::Sparse { idx_bytes, invert } => {
                let fill = if invert { 0xFFu8 } else { 0x00u8 };
                for slot in &mut scratch[..nbytes] {
                    slot.write(fill);
                }
                // SAFETY: Every byte in `scratch[..nbytes]` was initialized above.
                let dense = unsafe {
                    std::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<u8>(), nbytes)
                };
                for &[lo, hi] in idx_bytes.as_chunks::<2>().0 {
                    let pos = usize::from(u16::from_le_bytes([lo, hi]));
                    debug_assert!(pos < len);
                    if invert {
                        dense[pos >> 3] &= !(1u8 << (pos & 7));
                    } else {
                        dense[pos >> 3] |= 1u8 << (pos & 7);
                    }
                }
                if invert && (len & 7) != 0 && nbytes > 0 {
                    dense[nbytes - 1] &= (1u8 << (len & 7)) - 1;
                }
                DenseBitmap::Bytes(dense)
            }
            Self::Scratch => DenseBitmap::Bytes(scratch_as_init(scratch, nbytes)),
        }
    }
}

/// Returns `&scratch[..nbytes]` as an initialized byte slice after `NodeBitmap::Scratch` was decoded.
#[inline]
pub(crate) fn scratch_as_init(scratch: &[MaybeUninit<u8>], nbytes: usize) -> &[u8] {
    debug_assert!(scratch.len() >= nbytes);
    // SAFETY: Caller only invokes this for `NodeBitmap::Scratch`, where `decode_static_byte_rans`
    // initialized every byte in `0..nbytes`.
    unsafe { std::slice::from_raw_parts(scratch.as_ptr().cast::<u8>(), nbytes) }
}
