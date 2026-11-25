// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core 32-lane single-refill `Pow2-Bin + Raw-Bits` rANS encoder and decoder kernels.

#![expect(clippy::cast_possible_truncation)]

use std::cmp::Reverse;
use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_array::dtype::NativePType;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

/// Number of interleaved rANS state machines per block.
pub const NUM_LANES: usize = 32;

/// Maximum number of 32-element stripes per block.
pub const STRIPES_PER_BLOCK: usize = 128;

/// Number of logical rows per independent rANS block (`32 * 128 = 4,096`).
pub const BLOCK_VALUES: usize = NUM_LANES * STRIPES_PER_BLOCK;

/// Log2 of the normalized probability scale (`2^11 = 2,048`).
pub const PROB_BITS: u32 = 11;

/// Total normalized frequency scale (`2,048`).
pub const PROB_SCALE: u32 = 1 << PROB_BITS;

/// Bitmask for extracting the `11`-bit probability slot (`2,047`).
pub const PROB_MASK: u32 = PROB_SCALE - 1;

/// Lower bound of the rANS state invariant `x in [2^15, 2^31)`.
pub const RANS_L: u32 = 1 << 15;

/// Number of symbols in the `Pow2-Bin + Raw-Bits` alphabet (`0..=149`).
pub const MAX_SYMS: usize = 150;

/// Byte size of one serialized [`RansBlockHeader`] (`u32 + u16 + u16 = 8` bytes).
pub const BLOCK_HEADER_BYTES: usize = 8;

/// Byte size of the 32 initial `u32` rANS states stored at the start of each block (`128` bytes).
pub const BLOCK_STATES_BYTES: usize = NUM_LANES * 4;

/// Shift for 43-bit native `u64` division-free rANS encoding across `freq in 1..=2048`.
pub const RECIP_SHIFT: u32 = 43;

/// Maximum `ebits` where `(bit_pos & 7) + ebits <= 64` holds for every byte-unaligned `u64` load.
const FAST_EXTRA_MAX_EBITS: u8 = 57;

/// 43-bit reciprocal table for native `u64` division-free rANS encoding across `freq in 1..=2048`.
///
/// After renormalization `state < freq << 20`, so `u64::from(state) * RECIP43[freq] < 2^64` fits
/// without overflow in a single `u64` multiply and `(u64::from(state) * RECIP43[freq]) >> 43`
/// equals `state / freq` for every `freq in 1..=2048`.
pub static RECIP43: [u64; (PROB_SCALE as usize) + 1] = build_recip43();

const fn build_recip43() -> [u64; (PROB_SCALE as usize) + 1] {
    let mut table = [0u64; (PROB_SCALE as usize) + 1];
    table[1] = 1u64 << RECIP_SHIFT;
    let mut f = 2usize;
    while f <= PROB_SCALE as usize {
        table[f] = ((1u64 << RECIP_SHIFT) / (f as u64)) + 1;
        f += 1;
    }
    table
}

/// Base unsigned value for each symbol `0..150` in the `Pow2-Bin + Raw-Bits` bijection.
pub static BASE_TABLE: [u64; MAX_SYMS] = build_base_table();

/// Number of verbatim trailing raw bits (`0` or `4..=62`) for each symbol `0..150`.
pub static EBITS_TABLE: [u8; MAX_SYMS] = build_ebits_table();

const fn build_base_table() -> [u64; MAX_SYMS] {
    let mut table = [0u64; MAX_SYMS];
    let mut sym = 0usize;
    while sym < 32 {
        table[sym] = sym as u64;
        sym += 1;
    }
    while sym < MAX_SYMS {
        let rel = (sym - 32) as u8;
        let k = 5 + (rel >> 1);
        let ebits = k - 1;
        let msb2 = 2 + (rel & 1);
        table[sym] = (msb2 as u64) << ebits;
        sym += 1;
    }
    table
}

const fn build_ebits_table() -> [u8; MAX_SYMS] {
    let mut table = [0u8; MAX_SYMS];
    let mut sym = 32usize;
    while sym < MAX_SYMS {
        let rel = (sym - 32) as u8;
        let k = 5 + (rel >> 1);
        table[sym] = k - 1;
        sym += 1;
    }
    table
}

/// Decomposes `u` into `(sym, ebits, raw)` under the `Pow2-Bin + Raw-Bits` bijection.
#[inline]
pub const fn encode_pow2_bin(u: u64) -> (u8, u8, u64) {
    if u < 32 {
        (u as u8, 0, 0)
    } else {
        let k = (63 - u.leading_zeros()) as u8;
        let ebits = k - 1;
        let msb2 = ((u >> ebits) & 3) as u8;
        let sym = 32 + ((k - 5) << 1) + (msb2 - 2);
        let raw = u & ((1u64 << ebits) - 1);
        (sym, ebits, raw)
    }
}

/// Returns `(sym as usize, ebits)` for `u` without masking `raw`, used by fast histogram estimators.
#[inline]
pub const fn pow2_sym_and_ebits(u: u64) -> (usize, u8) {
    if u < 32 {
        (u as usize, 0)
    } else {
        let k = (63 - u.leading_zeros()) as u8;
        let ebits = k - 1;
        let msb2 = ((u >> ebits) & 3) as u8;
        let sym = 32 + ((k - 5) << 1) + (msb2 - 2);
        (sym as usize, ebits)
    }
}

/// 8-byte aligned decode lookup table entry (`2,048` entries = `16 KiB` L1-resident table).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeSlot {
    /// Normalized symbol frequency (`1..=2048`).
    pub freq: u16,
    /// `slot - cumfreq[sym]` (`0..freq`).
    pub bias: u16,
    /// Number of trailing raw bits for `sym` (`0` or `4..=62`).
    pub ebits: u8,
    /// Decoded symbol ID (`0..150`).
    pub sym: u8,
    /// Explicit padding to keep each entry at `8` bytes (`slot << 3` indexing).
    pub pad: u16,
}

/// Precomputed `2,048`-entry decode lookup table cached inside [`crate::RansData`].
#[derive(Clone, Debug)]
pub struct RansDecodeTable {
    /// Probability-slot lookup table indexed by `state & 2047`.
    pub slots: [DecodeSlot; PROB_SCALE as usize],
    /// Maximum `ebits` across all active symbols in this table (`0..=62`).
    pub max_ebits: u8,
}

impl RansDecodeTable {
    /// Builds and validates a [`RansDecodeTable`] from `(sym, freq)` pairs.
    ///
    /// Returns `(table, has_extra_bits)`.
    pub fn try_from_sym_freqs(sym_freqs: &[(u8, u16)]) -> VortexResult<(Self, bool)> {
        vortex_ensure!(
            !sym_freqs.is_empty() && sym_freqs.len() <= MAX_SYMS,
            "Rans sym_freqs length {} must be in 1..={MAX_SYMS}",
            sym_freqs.len()
        );

        let mut seen = [false; MAX_SYMS];
        let mut slots = [DecodeSlot::default(); PROB_SCALE as usize];
        let mut cum = 0usize;
        let mut max_ebits = 0u8;

        for &(sym, freq) in sym_freqs {
            let sym_idx = usize::from(sym);
            vortex_ensure!(
                sym_idx < MAX_SYMS,
                "Rans symbol {sym} exceeds maximum symbol {}",
                MAX_SYMS - 1
            );
            vortex_ensure!(!seen[sym_idx], "Duplicate Rans symbol {sym} in table");
            seen[sym_idx] = true;
            vortex_ensure!(
                freq >= 1 && u32::from(freq) <= PROB_SCALE,
                "Rans symbol {sym} has invalid frequency {freq}"
            );

            let ebits = EBITS_TABLE[sym_idx];
            max_ebits = max_ebits.max(ebits);
            let freq_usize = usize::from(freq);
            vortex_ensure!(
                cum + freq_usize <= PROB_SCALE as usize,
                "Rans cumulative frequency {} exceeds {PROB_SCALE}",
                cum + freq_usize
            );
            for rem in 0..freq_usize {
                slots[cum + rem] = DecodeSlot {
                    freq,
                    bias: rem as u16,
                    ebits,
                    sym,
                    pad: 0,
                };
            }
            cum += freq_usize;
        }

        vortex_ensure_eq!(
            cum,
            PROB_SCALE as usize,
            "Rans normalized frequencies must sum to {PROB_SCALE}"
        );
        let has_extra_bits = max_ebits > 0;
        Ok((Self { slots, max_ebits }, has_extra_bits))
    }
}

/// Encodes `(sym, freq)` pairs into compact 3-byte records `[sym, freq_lo, freq_hi]`.
pub fn pack_sym_freqs(sym_freqs: &[(u8, u16)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sym_freqs.len() * 3);
    for &(sym, freq) in sym_freqs {
        let [lo, hi] = freq.to_le_bytes();
        out.extend_from_slice(&[sym, lo, hi]);
    }
    out
}

/// Decodes compact 3-byte `[sym, freq_lo, freq_hi]` records from `bytes`.
pub fn unpack_sym_freqs(bytes: &[u8]) -> VortexResult<Vec<(u8, u16)>> {
    if !bytes.len().is_multiple_of(3) {
        vortex_bail!(
            "Rans sym_freqs byte length {} is not a multiple of 3",
            bytes.len()
        );
    }
    let count = bytes.len() / 3;
    vortex_ensure!(
        (1..=MAX_SYMS).contains(&count),
        "Rans sym_freqs count {count} must be in 1..={MAX_SYMS}"
    );
    let mut out = Vec::with_capacity(count);
    for &[sym, lo, hi] in bytes.as_chunks::<3>().0 {
        let freq = u16::from_le_bytes([lo, hi]);
        out.push((sym, freq));
    }
    Ok(out)
}

/// Normalizes non-zero symbol counts to sum to [`PROB_SCALE`] (`2048`) with every active symbol `>= 1`
/// using Hare-Niemeyer largest-remainder allocation.
pub fn normalize_counts(counts: &[u64; MAX_SYMS]) -> Vec<(u8, u16)> {
    let mut active: Vec<usize> = (0..MAX_SYMS).filter(|&sym| counts[sym] > 0).collect();
    if active.is_empty() {
        return vec![(0, PROB_SCALE as u16)];
    }
    if active.len() == 1 {
        return vec![(active[0] as u8, PROB_SCALE as u16)];
    }

    let total: u128 = active.iter().map(|&sym| u128::from(counts[sym])).sum();
    let scale = u128::from(PROB_SCALE);

    let mut freqs = [0u16; MAX_SYMS];
    let mut remainders = [0u128; MAX_SYMS];
    let mut floored_zero = [false; MAX_SYMS];
    let mut sum_freqs = 0i32;

    for &sym in &active {
        let scaled = u128::from(counts[sym]) * scale;
        let quot = (scaled / total) as u16;
        let rem = scaled % total;
        remainders[sym] = rem;
        if quot == 0 {
            freqs[sym] = 1;
            floored_zero[sym] = true;
        } else {
            freqs[sym] = quot;
        }
        sum_freqs += i32::from(freqs[sym]);
    }

    let target = PROB_SCALE as i32;
    if sum_freqs < target {
        let deficit = (target - sum_freqs) as usize;
        active.sort_by_key(|&sym| (floored_zero[sym], Reverse(remainders[sym]), sym));
        for &sym in active.iter().take(deficit) {
            freqs[sym] += 1;
        }
        active.sort_unstable();
    } else if sum_freqs > target {
        let mut surplus = (sum_freqs - target) as usize;
        let mut donors = active.clone();
        donors.sort_by_key(|&sym| (remainders[sym], Reverse(freqs[sym]), sym));
        while surplus > 0 {
            let mut progressed = false;
            for &sym in &donors {
                if surplus == 0 {
                    break;
                }
                if freqs[sym] > 1 {
                    freqs[sym] -= 1;
                    surplus -= 1;
                    progressed = true;
                }
            }
            debug_assert!(progressed, "active symbols <= 150 < 2048 guarantees donors");
        }
    }

    active
        .into_iter()
        .map(|sym| (sym as u8, freqs[sym]))
        .collect()
}

/// Compact 8-byte per-block header referencing slices inside `payload`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RansBlockHeader {
    /// Byte offset (prior to subtracting `payload_base_offset`) of `[u32; 32]` states + `u16` words.
    pub rans_byte_offset: u32,
    /// Number of `u16` refill words following the 128-byte initial state block (`<= 4,096`).
    pub rans_u16_len: u16,
    /// Number of `u64` words in the block's raw-bits stream (`0` when `!has_extra_bits`).
    pub raw_u64_len: u16,
}

impl RansBlockHeader {
    /// Reads block header `block_idx` from the host `block_headers` byte slice.
    #[inline]
    pub fn read_at(headers_bytes: &[u8], block_idx: usize) -> Self {
        let base = block_idx * BLOCK_HEADER_BYTES;
        let hdr = &headers_bytes[base..base + BLOCK_HEADER_BYTES];
        Self {
            rans_byte_offset: u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]),
            rans_u16_len: u16::from_le_bytes([hdr[4], hdr[5]]),
            raw_u64_len: u16::from_le_bytes([hdr[6], hdr[7]]),
        }
    }

    /// Computes the 8-byte-aligned start offset (prior to subtracting `payload_base_offset`) of
    /// this block's `raw_bits` stream.
    #[inline]
    pub fn raw_byte_offset(self) -> VortexResult<usize> {
        let rans_end = (self.rans_byte_offset as usize)
            .checked_add(BLOCK_STATES_BYTES + usize::from(self.rans_u16_len) * 2)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block raw_byte_offset overflow"))?;
        rans_end
            .checked_next_multiple_of(8)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block raw_byte_offset align8 overflow"))
    }

    /// Computes the end byte offset (prior to subtracting `payload_base_offset`) of this block.
    #[inline]
    pub fn end_byte_offset(self) -> VortexResult<usize> {
        self.raw_byte_offset()?
            .checked_add(usize::from(self.raw_u64_len) * 8)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block end_byte_offset overflow"))
    }

    #[inline]
    fn write_into(self, dst: &mut ByteBufferMut) {
        dst.extend_from_slice(&self.rans_byte_offset.to_le_bytes());
        dst.extend_from_slice(&self.rans_u16_len.to_le_bytes());
        dst.extend_from_slice(&self.raw_u64_len.to_le_bytes());
    }
}

#[derive(Clone, Copy, Default)]
struct EncodeSymInfo {
    freq: u32,
    cumfreq: u32,
    x_max: u32,
    recip: u64,
}

struct BitWriter {
    words: Vec<u64>,
    acc: u128,
    bits: u32,
}

impl BitWriter {
    fn with_capacity(cap_words: usize) -> Self {
        Self {
            words: Vec::with_capacity(cap_words),
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    fn write(&mut self, val: u64, nbits: u8) {
        if nbits == 0 {
            return;
        }
        self.acc |= u128::from(val) << self.bits;
        self.bits += u32::from(nbits);
        if self.bits >= 64 {
            self.words.push(self.acc as u64);
            self.acc >>= 64;
            self.bits -= 64;
        }
    }

    #[inline]
    fn finish_with_slack(&mut self, slack_words: usize) {
        if self.bits > 0 {
            self.words.push(self.acc as u64);
            self.acc = 0;
            self.bits = 0;
        }
        self.words.resize(self.words.len() + slack_words, 0u64);
    }

    fn clear(&mut self) {
        self.words.clear();
        self.acc = 0;
        self.bits = 0;
    }
}

/// Fallback `u128` reader used only when `max_ebits > 57` (`u64` values `>= 2^59`).
struct WideBitReader<'a> {
    chunks: &'a [[u8; 8]],
    pos: usize,
    acc: u128,
    bits: u32,
}

impl<'a> WideBitReader<'a> {
    #[inline]
    fn new(raw_bytes: &'a [u8]) -> Self {
        let chunks = raw_bytes.as_chunks::<8>().0;
        let mut acc = 0u128;
        let mut bits = 0u32;
        let mut pos = 0usize;
        if pos < chunks.len() {
            acc = u128::from(u64::from_le_bytes(chunks[pos]));
            bits = 64;
            pos += 1;
        }
        if pos < chunks.len() {
            acc |= u128::from(u64::from_le_bytes(chunks[pos])) << 64;
            bits = 128;
            pos += 1;
        }
        Self {
            chunks,
            pos,
            acc,
            bits,
        }
    }

    #[inline]
    fn read(&mut self, ebits: u8) -> u64 {
        let n = u32::from(ebits);
        let val = (self.acc as u64) & ((1u64 << n) - 1);
        self.acc >>= n;
        self.bits = self.bits.saturating_sub(n);
        if self.bits <= 64 && self.pos < self.chunks.len() {
            self.acc |=
                u128::from(u64::from_le_bytes(chunks_word(self.chunks, self.pos))) << self.bits;
            self.pos += 1;
            self.bits += 64;
        }
        val
    }
}

#[inline]
fn chunks_word(chunks: &[[u8; 8]], pos: usize) -> [u8; 8] {
    chunks[pos]
}

/// Output of encoding a slice of non-negative primitive integers with 32-lane `Pow2-Bin + Raw-Bits` rANS.
pub struct EncodedRansBuffers {
    /// Whether any active symbol uses extra raw bits (`sym >= 32`).
    pub has_extra_bits: bool,
    /// Normalized `(sym, freq)` table summing to `2048`.
    pub sym_freqs: Vec<(u8, u16)>,
    /// Precomputed decode lookup table.
    pub decode_table: RansDecodeTable,
    /// 8-byte aligned `8 * num_blocks` block headers buffer.
    pub block_headers: ByteBuffer,
    /// 8-byte aligned payload buffer containing block states, `u16` refills, and `u64` raw bits.
    pub payload: ByteBuffer,
}

/// Encodes non-negative `values` (`T` integer slice) with validity `mask` into 32-lane block-reset
/// rANS buffers. Signed slices containing negative valid values return an error (wrap with `ZigZag`).
pub fn encode_primitive_slice<T>(
    values: &[T],
    is_signed: bool,
    mask: &Mask,
) -> VortexResult<EncodedRansBuffers>
where
    T: NativePType + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let len = values.len();
    if len == 0 {
        let sym_freqs = vec![(0u8, PROB_SCALE as u16)];
        let (decode_table, has_extra_bits) = RansDecodeTable::try_from_sym_freqs(&sym_freqs)?;
        return Ok(EncodedRansBuffers {
            has_extra_bits,
            sym_freqs,
            decode_table,
            block_headers: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>()).freeze(),
            payload: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>()).freeze(),
        });
    }

    let (counts, fill_sym) = count_chunk_symbols(values, is_signed, mask)?;
    let sym_freqs = normalize_counts(&counts);
    let (decode_table, has_extra_bits) = RansDecodeTable::try_from_sym_freqs(&sym_freqs)?;
    let slack_words = if !has_extra_bits {
        0
    } else if decode_table.max_ebits <= FAST_EXTRA_MAX_EBITS {
        1
    } else {
        2
    };

    let mut enc_table = [EncodeSymInfo::default(); MAX_SYMS];
    let mut cum = 0u32;
    for &(sym, freq) in &sym_freqs {
        let f = u32::from(freq);
        enc_table[usize::from(sym)] = EncodeSymInfo {
            freq: f,
            cumfreq: cum,
            x_max: f << 20,
            recip: RECIP43[f as usize],
        };
        cum += f;
    }

    let num_blocks = len.div_ceil(BLOCK_VALUES);
    let mut headers_buf = ByteBufferMut::with_capacity_aligned(
        num_blocks * BLOCK_HEADER_BYTES,
        Alignment::of::<u64>(),
    );
    let mut payload_buf =
        ByteBufferMut::with_capacity_aligned(num_blocks * 256, Alignment::of::<u64>());

    let mut scratch = BlockEncodeScratch {
        block_syms: [fill_sym; BLOCK_VALUES],
        emitted_u16: [0u16; BLOCK_VALUES],
        bit_writer: BitWriter::with_capacity(BLOCK_VALUES),
    };

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_VALUES;
        let block_end = (block_start + BLOCK_VALUES).min(len);
        encode_one_block(
            values,
            block_start,
            block_end,
            is_signed,
            has_extra_bits,
            slack_words,
            fill_sym,
            mask,
            &enc_table,
            &mut scratch,
            &mut headers_buf,
            &mut payload_buf,
        )?;
    }

    Ok(EncodedRansBuffers {
        has_extra_bits,
        sym_freqs,
        decode_table,
        block_headers: headers_buf.freeze(),
        payload: payload_buf.freeze(),
    })
}

#[inline]
fn nonneg_to_u64<T>(val: T, is_signed: bool) -> u64
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    if is_signed {
        AsPrimitive::<i64>::as_(val) as u64
    } else {
        AsPrimitive::<u64>::as_(val)
    }
}

fn count_chunk_symbols<T>(
    values: &[T],
    is_signed: bool,
    mask: &Mask,
) -> VortexResult<([u64; MAX_SYMS], u8)>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let len = values.len();
    let mut counts = [0u64; MAX_SYMS];
    let mut valid_count = 0u64;

    if matches!(mask, Mask::AllTrue(_)) {
        if is_signed {
            for &v in values {
                let signed = AsPrimitive::<i64>::as_(v);
                vortex_ensure!(
                    signed >= 0,
                    "Rans::encode requires non-negative integer values (got {signed}); wrap negative arrays with ZigZag"
                );
                counts[pow2_sym_and_ebits(signed as u64).0] += 1;
            }
        } else {
            for &v in values {
                counts[pow2_sym_and_ebits(AsPrimitive::<u64>::as_(v)).0] += 1;
            }
        }
        valid_count = len as u64;
    } else if let vortex_mask::AllOr::Some(buf) = mask.bit_buffer() {
        let mut neg_found: Option<i64> = None;
        buf.for_each_set_index(|idx| {
            if is_signed {
                let signed = AsPrimitive::<i64>::as_(values[idx]);
                if signed < 0 {
                    neg_found = Some(signed);
                    return;
                }
                counts[pow2_sym_and_ebits(signed as u64).0] += 1;
            } else {
                counts[pow2_sym_and_ebits(AsPrimitive::<u64>::as_(values[idx])).0] += 1;
            }
            valid_count += 1;
        });
        if let Some(neg) = neg_found {
            vortex_bail!(
                "Rans::encode requires non-negative integer values (got {neg}); wrap negative arrays with ZigZag"
            );
        }
    }

    let mut fill_sym = 0u8;
    let mut best_exact_count = 0u64;
    for (sym, &cnt) in counts[..32].iter().enumerate() {
        if cnt > best_exact_count {
            best_exact_count = cnt;
            fill_sym = sym as u8;
        }
    }

    let num_blocks = len.div_ceil(BLOCK_VALUES);
    let last_block_len = len - (num_blocks - 1) * BLOCK_VALUES;
    let padded_total =
        (num_blocks - 1) * BLOCK_VALUES + last_block_len.div_ceil(NUM_LANES) * NUM_LANES;
    let fill_positions = (padded_total as u64).saturating_sub(valid_count);
    counts[usize::from(fill_sym)] += fill_positions;

    Ok((counts, fill_sym))
}

struct BlockEncodeScratch {
    block_syms: [u8; BLOCK_VALUES],
    emitted_u16: [u16; BLOCK_VALUES],
    bit_writer: BitWriter,
}

#[expect(clippy::too_many_arguments)]
fn encode_one_block<T>(
    values: &[T],
    block_start: usize,
    block_end: usize,
    is_signed: bool,
    has_extra_bits: bool,
    slack_words: usize,
    fill_sym: u8,
    mask: &Mask,
    enc_table: &[EncodeSymInfo; MAX_SYMS],
    scratch: &mut BlockEncodeScratch,
    headers_buf: &mut ByteBufferMut,
    payload_buf: &mut ByteBufferMut,
) -> VortexResult<()>
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let block_len = block_end - block_start;
    let n_stripes = block_len.div_ceil(NUM_LANES);
    let padded_len = n_stripes * NUM_LANES;
    scratch.bit_writer.clear();

    populate_block_symbols(
        &values[block_start..block_end],
        block_start,
        padded_len,
        is_signed,
        has_extra_bits,
        slack_words,
        fill_sym,
        mask,
        &mut scratch.block_syms,
        &mut scratch.bit_writer,
    );

    let mut states = [RANS_L; NUM_LANES];
    let mut emit_pos = BLOCK_VALUES;
    for stripe in (0..n_stripes).rev() {
        let stripe_base = stripe * NUM_LANES;
        for lane in (0..NUM_LANES).rev() {
            let sym = scratch.block_syms[stripe_base + lane];
            let info = enc_table[usize::from(sym)];
            let mut state = states[lane];
            if state >= info.x_max {
                emit_pos -= 1;
                scratch.emitted_u16[emit_pos] = state as u16;
                state >>= 16;
            }
            let quot = ((u64::from(state).wrapping_mul(info.recip)) >> RECIP_SHIFT) as u32;
            let rem = state - quot * info.freq;
            states[lane] = (quot << PROB_BITS) + info.cumfreq + rem;
        }
    }
    let emitted = &scratch.emitted_u16[emit_pos..];

    let rans_byte_offset = u32::try_from(payload_buf.len())?;
    let rans_u16_len = u16::try_from(emitted.len())?;
    append_u32_le_slice(payload_buf, &states);
    append_u16_le_slice(payload_buf, emitted);
    while !payload_buf.len().is_multiple_of(8) {
        payload_buf.push(0);
    }

    let raw_u64_len = u16::try_from(scratch.bit_writer.words.len())?;
    append_u64_le_slice(payload_buf, &scratch.bit_writer.words);

    RansBlockHeader {
        rans_byte_offset,
        rans_u16_len,
        raw_u64_len,
    }
    .write_into(headers_buf);
    Ok(())
}

#[inline]
fn append_u32_le_slice(dst: &mut ByteBufferMut, slice: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `u32` has no uninitialized padding bytes and host is little-endian.
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        dst.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &w in slice {
            dst.extend_from_slice(&w.to_le_bytes());
        }
    }
}

#[inline]
fn append_u16_le_slice(dst: &mut ByteBufferMut, slice: &[u16]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `u16` has no uninitialized padding bytes and host is little-endian.
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        dst.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &w in slice {
            dst.extend_from_slice(&w.to_le_bytes());
        }
    }
}

#[inline]
fn append_u64_le_slice(dst: &mut ByteBufferMut, slice: &[u64]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `u64` has no uninitialized padding bytes and host is little-endian.
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        dst.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &w in slice {
            dst.extend_from_slice(&w.to_le_bytes());
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn populate_block_symbols<T>(
    block_vals: &[T],
    block_start: usize,
    padded_len: usize,
    is_signed: bool,
    has_extra_bits: bool,
    slack_words: usize,
    fill_sym: u8,
    mask: &Mask,
    block_syms: &mut [u8; BLOCK_VALUES],
    bit_writer: &mut BitWriter,
) where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let block_len = block_vals.len();
    match mask {
        Mask::AllFalse(_) => {
            block_syms[..padded_len].fill(fill_sym);
            if has_extra_bits {
                bit_writer.finish_with_slack(slack_words);
            }
        }
        Mask::AllTrue(_) if has_extra_bits => {
            for (i, &v) in block_vals.iter().enumerate() {
                let (sym, ebits, raw) = encode_pow2_bin(nonneg_to_u64(v, is_signed));
                block_syms[i] = sym;
                bit_writer.write(raw, ebits);
            }
            block_syms[block_len..padded_len].fill(fill_sym);
            bit_writer.finish_with_slack(slack_words);
        }
        Mask::AllTrue(_) => {
            for (i, &v) in block_vals.iter().enumerate() {
                block_syms[i] = nonneg_to_u64(v, is_signed) as u8;
            }
            block_syms[block_len..padded_len].fill(fill_sym);
        }
        Mask::Values(mv) if has_extra_bits => {
            let bits = mv.bit_buffer();
            for (i, &v) in block_vals.iter().enumerate() {
                if bits.value(block_start + i) {
                    let (sym, ebits, raw) = encode_pow2_bin(nonneg_to_u64(v, is_signed));
                    block_syms[i] = sym;
                    bit_writer.write(raw, ebits);
                } else {
                    block_syms[i] = fill_sym;
                }
            }
            block_syms[block_len..padded_len].fill(fill_sym);
            bit_writer.finish_with_slack(slack_words);
        }
        Mask::Values(mv) => {
            let bits = mv.bit_buffer();
            for (i, &v) in block_vals.iter().enumerate() {
                block_syms[i] = if bits.value(block_start + i) {
                    nonneg_to_u64(v, is_signed) as u8
                } else {
                    fill_sym
                };
            }
            block_syms[block_len..padded_len].fill(fill_sym);
        }
    }
}

/// Validates that all `num_blocks` headers in `headers_bytes` are monotonically ordered and lie
/// cleanly within `payload_len` relative to `payload_base_offset`.
pub fn validate_block_headers(
    headers_bytes: &[u8],
    payload_len: usize,
    payload_base_offset: u32,
    num_blocks: usize,
    has_extra_bits: bool,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        headers_bytes.len(),
        num_blocks * BLOCK_HEADER_BYTES,
        "Rans block_headers byte length mismatch"
    );
    let base = payload_base_offset as usize;
    let mut prev_end = 0usize;
    for b in 0..num_blocks {
        let hdr = RansBlockHeader::read_at(headers_bytes, b);
        let rans_start = (hdr.rans_byte_offset as usize)
            .checked_sub(base)
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    "Rans block {b} rans_byte_offset {} precedes payload_base_offset {base}",
                    hdr.rans_byte_offset
                )
            })?;
        vortex_ensure!(
            prev_end <= rans_start,
            "Rans block {b} rans_start {rans_start} overlaps previous block end {prev_end}"
        );
        let raw_start = hdr
            .raw_byte_offset()?
            .checked_sub(base)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} raw_byte_offset underflow"))?;
        if !has_extra_bits {
            vortex_ensure_eq!(
                hdr.raw_u64_len,
                0,
                "Rans block {b} has raw_u64_len {} when has_extra_bits is false",
                hdr.raw_u64_len
            );
        }
        let raw_end = hdr
            .end_byte_offset()?
            .checked_sub(base)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} end_byte_offset underflow"))?;
        vortex_ensure!(
            raw_end <= payload_len,
            "Rans block {b} end {raw_end} exceeds payload length {payload_len}"
        );
        debug_assert!(rans_start < raw_start && raw_start <= raw_end);
        prev_end = raw_end;
    }
    Ok(())
}

/// Decodes the first `n_stripes` (`1..=128`) of `block_idx` into uninitialized `dst[..n_stripes * 32]`
/// applying `map_out` to each decoded `u64` value in register.
#[expect(clippy::too_many_arguments)]
#[inline]
pub fn decode_block_stripes_into<Out, F>(
    table: &RansDecodeTable,
    headers_bytes: &[u8],
    payload_bytes: &[u8],
    payload_base_offset: u32,
    block_idx: usize,
    n_stripes: usize,
    has_extra_bits: bool,
    dst: &mut [MaybeUninit<Out>],
    map_out: F,
) where
    Out: Copy,
    F: Copy + Fn(u64) -> Out,
{
    debug_assert!(n_stripes <= STRIPES_PER_BLOCK);
    debug_assert!(dst.len() >= n_stripes * NUM_LANES);

    let hdr = RansBlockHeader::read_at(headers_bytes, block_idx);
    let base = payload_base_offset as usize;
    let rans_start = (hdr.rans_byte_offset as usize) - base;
    let state_bytes = &payload_bytes[rans_start..rans_start + BLOCK_STATES_BYTES];
    let mut states = [0u32; NUM_LANES];
    for (lane, chunk) in states.iter_mut().zip(state_bytes.as_chunks::<4>().0.iter()) {
        *lane = u32::from_le_bytes(*chunk);
    }

    let words_start = rans_start + BLOCK_STATES_BYTES;
    let words_end = words_start + usize::from(hdr.rans_u16_len) * 2;
    let word_chunks = payload_bytes[words_start..words_end].as_chunks::<2>().0;

    if !has_extra_bits {
        decode_stripes_no_extra(table, &mut states, word_chunks, n_stripes, dst, map_out);
    } else {
        let raw_start = words_end.next_multiple_of(8);
        let raw_end = raw_start + usize::from(hdr.raw_u64_len) * 8;
        let raw_bytes = &payload_bytes[raw_start..raw_end];
        if table.max_ebits <= FAST_EXTRA_MAX_EBITS {
            decode_stripes_with_fast_extra(
                table,
                &mut states,
                word_chunks,
                raw_bytes,
                n_stripes,
                dst,
                map_out,
            );
        } else {
            decode_stripes_with_wide_extra(
                table,
                &mut states,
                word_chunks,
                raw_bytes,
                n_stripes,
                dst,
                map_out,
            );
        }
    }
}

/// Convenience wrapper decoding `u64` values cast directly to `T`.
#[expect(clippy::too_many_arguments)]
#[inline]
pub fn decode_block_stripes<T>(
    table: &RansDecodeTable,
    headers_bytes: &[u8],
    payload_bytes: &[u8],
    payload_base_offset: u32,
    block_idx: usize,
    n_stripes: usize,
    has_extra_bits: bool,
    dst: &mut [MaybeUninit<T>],
) where
    T: NativePType,
    u64: AsPrimitive<T>,
{
    decode_block_stripes_into(
        table,
        headers_bytes,
        payload_bytes,
        payload_base_offset,
        block_idx,
        n_stripes,
        has_extra_bits,
        dst,
        |u| u.as_(),
    );
}

#[inline]
fn refill_lanes(
    states: &mut [u32; NUM_LANES],
    next_states: &[u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    word_pos: &mut usize,
) {
    let mut pos = *word_pos;
    if pos + NUM_LANES <= word_chunks.len() {
        for lane in 0..NUM_LANES {
            let s = next_states[lane];
            let need = s < RANS_L;
            // SAFETY: `pos` starts `<= word_chunks.len() - 32` and increments at most once per
            // lane across `0..32`, so `pos < word_chunks.len()` holds for every lane.
            let word = u32::from(u16::from_le_bytes(*unsafe {
                word_chunks.get_unchecked(pos)
            }));
            states[lane] = if need { (s << 16) | word } else { s };
            pos += usize::from(need);
        }
    } else {
        for lane in 0..NUM_LANES {
            let mut s = next_states[lane];
            if s < RANS_L && pos < word_chunks.len() {
                let word = u32::from(u16::from_le_bytes(word_chunks[pos]));
                pos += 1;
                s = (s << 16) | word;
            }
            states[lane] = s;
        }
    }
    *word_pos = pos;
}

/// Fast path when `has_extra_bits == false`: pure 32-lane state update + branchless `u16` refill.
fn decode_stripes_no_extra<Out, F>(
    table: &RansDecodeTable,
    states: &mut [u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    n_stripes: usize,
    dst: &mut [MaybeUninit<Out>],
    map_out: F,
) where
    Out: Copy,
    F: Copy + Fn(u64) -> Out,
{
    let mut word_pos = 0usize;
    let mut syms = [0u8; NUM_LANES];
    let mut next_states = [0u32; NUM_LANES];

    for stripe in 0..n_stripes {
        for lane in 0..NUM_LANES {
            let state = states[lane];
            let slot = (state & PROB_MASK) as usize;
            let entry = table.slots[slot];
            syms[lane] = entry.sym;
            next_states[lane] =
                u32::from(entry.freq) * (state >> PROB_BITS) + u32::from(entry.bias);
        }

        refill_lanes(states, &next_states, word_chunks, &mut word_pos);

        let out_stripe = &mut dst[stripe * NUM_LANES..(stripe + 1) * NUM_LANES];
        for lane in 0..NUM_LANES {
            out_stripe[lane].write(map_out(u64::from(syms[lane])));
        }
    }
}

/// Branchless parallel `raw_bits` path for `max_ebits <= 57` (all `u8..=u32`, `i8..=i32`, and
/// `u64`/`i64` values `< 2^59`): prefix-sums `ebits_arr` across the 32 lanes so every lane loads
/// its `u64` raw-bit window independently with zero serial bitstream dependency or branches.
fn decode_stripes_with_fast_extra<Out, F>(
    table: &RansDecodeTable,
    states: &mut [u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    raw_bytes: &[u8],
    n_stripes: usize,
    dst: &mut [MaybeUninit<Out>],
    map_out: F,
) where
    Out: Copy,
    F: Copy + Fn(u64) -> Out,
{
    let mut word_pos = 0usize;
    let mut bit_cursor = 0usize;
    let mut syms = [0u8; NUM_LANES];
    let mut ebits_arr = [0u8; NUM_LANES];
    let mut bit_offsets = [0usize; NUM_LANES];
    let mut next_states = [0u32; NUM_LANES];
    let raw_ptr = raw_bytes.as_ptr();
    let raw_len = raw_bytes.len();

    for stripe in 0..n_stripes {
        for lane in 0..NUM_LANES {
            let state = states[lane];
            let slot = (state & PROB_MASK) as usize;
            let entry = table.slots[slot];
            syms[lane] = entry.sym;
            ebits_arr[lane] = entry.ebits;
            next_states[lane] =
                u32::from(entry.freq) * (state >> PROB_BITS) + u32::from(entry.bias);
        }

        refill_lanes(states, &next_states, word_chunks, &mut word_pos);

        for lane in 0..NUM_LANES {
            bit_offsets[lane] = bit_cursor;
            bit_cursor += usize::from(ebits_arr[lane]);
        }

        let out_stripe = &mut dst[stripe * NUM_LANES..(stripe + 1) * NUM_LANES];
        if (bit_cursor >> 3) + 8 <= raw_len {
            for lane in 0..NUM_LANES {
                let bp = bit_offsets[lane];
                // SAFETY: `bp <= bit_cursor`, so `(bp >> 3) + 8 <= (bit_cursor >> 3) + 8 <= raw_len`.
                let word = unsafe { raw_ptr.add(bp >> 3).cast::<u64>().read_unaligned() }.to_le();
                let shift = (bp & 7) as u32;
                let ebits = u32::from(ebits_arr[lane]);
                let mask = (1u64 << ebits) - 1;
                let raw = (word >> shift) & mask;
                let u = BASE_TABLE[usize::from(syms[lane])] | raw;
                out_stripe[lane].write(map_out(u));
            }
        } else {
            for lane in 0..NUM_LANES {
                let bp = bit_offsets[lane];
                let byte_idx = bp >> 3;
                let mut buf = [0u8; 8];
                if byte_idx < raw_len {
                    let avail = (raw_len - byte_idx).min(8);
                    buf[..avail].copy_from_slice(&raw_bytes[byte_idx..byte_idx + avail]);
                }
                let word = u64::from_le_bytes(buf);
                let shift = (bp & 7) as u32;
                let ebits = u32::from(ebits_arr[lane]);
                let raw = (word >> shift) & ((1u64 << ebits) - 1);
                let u = BASE_TABLE[usize::from(syms[lane])] | raw;
                out_stripe[lane].write(map_out(u));
            }
        }
    }
}

/// Wide `u128` fallback when `max_ebits > 57` (`u64` values `>= 2^59`).
fn decode_stripes_with_wide_extra<Out, F>(
    table: &RansDecodeTable,
    states: &mut [u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    raw_bytes: &[u8],
    n_stripes: usize,
    dst: &mut [MaybeUninit<Out>],
    map_out: F,
) where
    Out: Copy,
    F: Copy + Fn(u64) -> Out,
{
    let mut word_pos = 0usize;
    let mut bit_reader = WideBitReader::new(raw_bytes);
    let mut syms = [0u8; NUM_LANES];
    let mut ebits_arr = [0u8; NUM_LANES];
    let mut next_states = [0u32; NUM_LANES];

    for stripe in 0..n_stripes {
        for lane in 0..NUM_LANES {
            let state = states[lane];
            let slot = (state & PROB_MASK) as usize;
            let entry = table.slots[slot];
            syms[lane] = entry.sym;
            ebits_arr[lane] = entry.ebits;
            next_states[lane] =
                u32::from(entry.freq) * (state >> PROB_BITS) + u32::from(entry.bias);
        }

        refill_lanes(states, &next_states, word_chunks, &mut word_pos);

        let out_stripe = &mut dst[stripe * NUM_LANES..(stripe + 1) * NUM_LANES];
        for lane in 0..NUM_LANES {
            let base = BASE_TABLE[usize::from(syms[lane])];
            let raw = bit_reader.read(ebits_arr[lane]);
            out_stripe[lane].write(map_out(base | raw));
        }
    }
}
