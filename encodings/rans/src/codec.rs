// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core 32-lane single-refill `Pow2-Bin + Raw-Bits` rANS encoder and decoder kernels.

#![expect(clippy::cast_possible_truncation)]

use std::cmp::Reverse;

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

/// Byte size of one serialized [`RansBlockHeader`] (`4 * u32 = 16` bytes).
pub const BLOCK_HEADER_BYTES: usize = 16;

/// Byte size of the 32 initial `u32` rANS states stored at the start of each block (`128` bytes).
pub const BLOCK_STATES_BYTES: usize = NUM_LANES * 4;

/// 48-bit reciprocal table for division-free rANS encoding across `freq in 1..=2048`.
///
/// For `x < 2^31` and `f in 1..=2048`, `q = ((x as u128 * RECIP48[f] as u128) >> 48) as u32`
/// equals `x / f` with zero hardware `udiv`.
pub static RECIP48: [u64; (PROB_SCALE as usize) + 1] = build_recip48();

const fn build_recip48() -> [u64; (PROB_SCALE as usize) + 1] {
    let mut table = [0u64; (PROB_SCALE as usize) + 1];
    table[1] = 1u64 << 48;
    let mut f = 2usize;
    while f <= PROB_SCALE as usize {
        table[f] = ((1u64 << 48) / (f as u64)) + 1;
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

/// ZigZag-encodes a signed 64-bit integer into `u64` (`0 -> 0, -1 -> 1, 1 -> 2, i64::MIN -> u64::MAX`).
#[inline]
pub const fn zigzag_encode(val: i64) -> u64 {
    ((val as u64) << 1) ^ ((val >> 63) as u64)
}

/// Inverse ZigZag mapping from `u64` back to `i64`, valid across the full `i64::MIN..=i64::MAX` range.
#[inline]
pub const fn zigzag_decode(val: u64) -> i64 {
    ((val >> 1) as i64) ^ -((val & 1) as i64)
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
        let mut has_extra_bits = false;

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
            if ebits > 0 {
                has_extra_bits = true;
            }
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
        Ok((Self { slots }, has_extra_bits))
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
        // Award +1 to symbols that were not already rounded up from 0, ordered by largest remainder.
        active.sort_by_key(|&sym| (floored_zero[sym], Reverse(remainders[sym]), sym));
        for &sym in active.iter().take(deficit) {
            freqs[sym] += 1;
        }
        active.sort_unstable();
    } else if sum_freqs > target {
        let mut surplus = (sum_freqs - target) as usize;
        // Reclaim surplus from symbols with freq > 1, smallest remainder first (breaking ties by
        // largest frequency so relative distortion stays minimal).
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

/// Parsed 16-byte per-block header referencing slices inside `payload`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RansBlockHeader {
    /// Byte offset (prior to subtracting `payload_base_offset`) of `[u32; 32]` states + `u16` words.
    pub rans_byte_offset: u32,
    /// Number of `u16` refill words following the 128-byte initial state block.
    pub rans_u16_len: u32,
    /// Byte offset (prior to subtracting `payload_base_offset`) of packed `u64` raw-bit words.
    pub raw_byte_offset: u32,
    /// Number of `u64` words in the block's raw-bits stream.
    pub raw_u64_len: u32,
}

impl RansBlockHeader {
    /// Reads block header `block_idx` from the host `block_headers` byte slice.
    #[inline]
    pub fn read_at(headers_bytes: &[u8], block_idx: usize) -> Self {
        let base = block_idx * BLOCK_HEADER_BYTES;
        let hdr = &headers_bytes[base..base + BLOCK_HEADER_BYTES];
        Self {
            rans_byte_offset: u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]),
            rans_u16_len: u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]),
            raw_byte_offset: u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]),
            raw_u64_len: u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]),
        }
    }

    #[inline]
    fn write_into(self, dst: &mut ByteBufferMut) {
        dst.extend_from_slice(&self.rans_byte_offset.to_le_bytes());
        dst.extend_from_slice(&self.rans_u16_len.to_le_bytes());
        dst.extend_from_slice(&self.raw_byte_offset.to_le_bytes());
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
    fn finish(&mut self) {
        if self.bits > 0 {
            self.words.push(self.acc as u64);
            self.acc = 0;
            self.bits = 0;
        }
    }

    fn clear(&mut self) {
        self.words.clear();
        self.acc = 0;
        self.bits = 0;
    }
}

struct BitReader<'a> {
    chunks: &'a [[u8; 8]],
    pos: usize,
    acc: u128,
    bits: u32,
}

impl<'a> BitReader<'a> {
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
            self.acc |= u128::from(u64::from_le_bytes(self.chunks[self.pos])) << self.bits;
            self.pos += 1;
            self.bits += 64;
        }
        val
    }
}

/// Output of encoding a slice of primitive integers with 32-lane `Pow2-Bin + Raw-Bits` rANS.
pub struct EncodedRansBuffers {
    /// Whether signed ZigZag mapping was applied.
    pub signed_zigzag: bool,
    /// Whether any active symbol uses extra raw bits (`sym >= 32`).
    pub has_extra_bits: bool,
    /// Normalized `(sym, freq)` table summing to `2048`.
    pub sym_freqs: Vec<(u8, u16)>,
    /// Precomputed decode lookup table.
    pub decode_table: RansDecodeTable,
    /// 8-byte aligned `16 * num_blocks` block headers buffer.
    pub block_headers: ByteBuffer,
    /// 8-byte aligned payload buffer containing block states, `u16` refills, and `u64` raw bits.
    pub payload: ByteBuffer,
}

/// Encodes `values` (`T` integer slice) with validity `mask` into 32-lane block-reset rANS buffers.
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
            signed_zigzag: false,
            has_extra_bits,
            sym_freqs,
            decode_table,
            block_headers: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>()).freeze(),
            payload: ByteBufferMut::with_capacity_aligned(0, Alignment::of::<u64>()).freeze(),
        });
    }

    let signed_zigzag = detect_signed_zigzag(values, is_signed, mask);
    let (counts, fill_sym) = count_chunk_symbols(values, is_signed, signed_zigzag, mask);

    let sym_freqs = normalize_counts(&counts);
    let (decode_table, has_extra_bits) = RansDecodeTable::try_from_sym_freqs(&sym_freqs)?;

    let mut enc_table = [EncodeSymInfo::default(); MAX_SYMS];
    let mut cum = 0u32;
    for &(sym, freq) in &sym_freqs {
        let f = u32::from(freq);
        enc_table[usize::from(sym)] = EncodeSymInfo {
            freq: f,
            cumfreq: cum,
            x_max: f << 20,
            recip: RECIP48[f as usize],
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
        emitted_u16: Vec::with_capacity(BLOCK_VALUES),
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
            signed_zigzag,
            has_extra_bits,
            fill_sym,
            mask,
            &enc_table,
            &mut scratch,
            &mut headers_buf,
            &mut payload_buf,
        )?;
    }

    Ok(EncodedRansBuffers {
        signed_zigzag,
        has_extra_bits,
        sym_freqs,
        decode_table,
        block_headers: headers_buf.freeze(),
        payload: payload_buf.freeze(),
    })
}

#[inline]
fn map_int_to_u64<T>(val: T, is_signed: bool, signed_zigzag: bool) -> u64
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    if signed_zigzag {
        zigzag_encode(AsPrimitive::<i64>::as_(val))
    } else if is_signed {
        AsPrimitive::<i64>::as_(val) as u64
    } else {
        AsPrimitive::<u64>::as_(val)
    }
}

fn detect_signed_zigzag<T>(values: &[T], is_signed: bool, mask: &Mask) -> bool
where
    T: Copy + AsPrimitive<i64>,
{
    if !is_signed || matches!(mask, Mask::AllFalse(_)) {
        return false;
    }
    if matches!(mask, Mask::AllTrue(_)) {
        return values.iter().any(|&v| AsPrimitive::<i64>::as_(v) < 0);
    }
    let mut has_neg = false;
    if let vortex_mask::AllOr::Some(buf) = mask.bit_buffer() {
        buf.for_each_set_index(|idx| {
            if AsPrimitive::<i64>::as_(values[idx]) < 0 {
                has_neg = true;
            }
        });
    }
    has_neg
}

fn count_chunk_symbols<T>(
    values: &[T],
    is_signed: bool,
    signed_zigzag: bool,
    mask: &Mask,
) -> ([u64; MAX_SYMS], u8)
where
    T: Copy + AsPrimitive<i64> + AsPrimitive<u64>,
{
    let len = values.len();
    let mut counts = [0u64; MAX_SYMS];
    let mut valid_count = 0u64;

    if matches!(mask, Mask::AllTrue(_)) {
        for &v in values {
            let (sym, _) = pow2_sym_and_ebits(map_int_to_u64(v, is_signed, signed_zigzag));
            counts[sym] += 1;
        }
        valid_count = len as u64;
    } else if let vortex_mask::AllOr::Some(buf) = mask.bit_buffer() {
        buf.for_each_set_index(|idx| {
            let (sym, _) =
                pow2_sym_and_ebits(map_int_to_u64(values[idx], is_signed, signed_zigzag));
            counts[sym] += 1;
            valid_count += 1;
        });
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

    (counts, fill_sym)
}

struct BlockEncodeScratch {
    block_syms: [u8; BLOCK_VALUES],
    emitted_u16: Vec<u16>,
    bit_writer: BitWriter,
}

#[expect(clippy::too_many_arguments)]
fn encode_one_block<T>(
    values: &[T],
    block_start: usize,
    block_end: usize,
    is_signed: bool,
    signed_zigzag: bool,
    has_extra_bits: bool,
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
        signed_zigzag,
        has_extra_bits,
        fill_sym,
        mask,
        &mut scratch.block_syms,
        &mut scratch.bit_writer,
    );

    let mut states = [RANS_L; NUM_LANES];
    scratch.emitted_u16.clear();
    for stripe in (0..n_stripes).rev() {
        let stripe_base = stripe * NUM_LANES;
        for lane in (0..NUM_LANES).rev() {
            let sym = scratch.block_syms[stripe_base + lane];
            let info = enc_table[usize::from(sym)];
            let mut state = states[lane];
            if state >= info.x_max {
                scratch.emitted_u16.push(state as u16);
                state >>= 16;
            }
            let quot = ((u128::from(state) * u128::from(info.recip)) >> 48) as u32;
            let rem = state - quot * info.freq;
            states[lane] = (quot << PROB_BITS) + info.cumfreq + rem;
        }
    }
    scratch.emitted_u16.reverse();

    let rans_byte_offset = u32::try_from(payload_buf.len())?;
    let rans_u16_len = u32::try_from(scratch.emitted_u16.len())?;
    for state in states {
        payload_buf.extend_from_slice(&state.to_le_bytes());
    }
    for &word in &scratch.emitted_u16 {
        payload_buf.extend_from_slice(&word.to_le_bytes());
    }
    while !payload_buf.len().is_multiple_of(8) {
        payload_buf.push(0);
    }

    let raw_byte_offset = u32::try_from(payload_buf.len())?;
    let raw_u64_len = u32::try_from(scratch.bit_writer.words.len())?;
    for &word in &scratch.bit_writer.words {
        payload_buf.extend_from_slice(&word.to_le_bytes());
    }

    RansBlockHeader {
        rans_byte_offset,
        rans_u16_len,
        raw_byte_offset,
        raw_u64_len,
    }
    .write_into(headers_buf);
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn populate_block_symbols<T>(
    block_vals: &[T],
    block_start: usize,
    padded_len: usize,
    is_signed: bool,
    signed_zigzag: bool,
    has_extra_bits: bool,
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
        }
        Mask::AllTrue(_) if has_extra_bits => {
            for (i, &v) in block_vals.iter().enumerate() {
                let (sym, ebits, raw) =
                    encode_pow2_bin(map_int_to_u64(v, is_signed, signed_zigzag));
                block_syms[i] = sym;
                bit_writer.write(raw, ebits);
            }
            block_syms[block_len..padded_len].fill(fill_sym);
            bit_writer.finish();
        }
        Mask::AllTrue(_) => {
            for (i, &v) in block_vals.iter().enumerate() {
                block_syms[i] = map_int_to_u64(v, is_signed, signed_zigzag) as u8;
            }
            block_syms[block_len..padded_len].fill(fill_sym);
        }
        Mask::Values(mv) if has_extra_bits => {
            let bits = mv.bit_buffer();
            for (i, &v) in block_vals.iter().enumerate() {
                if bits.value(block_start + i) {
                    let (sym, ebits, raw) =
                        encode_pow2_bin(map_int_to_u64(v, is_signed, signed_zigzag));
                    block_syms[i] = sym;
                    bit_writer.write(raw, ebits);
                } else {
                    block_syms[i] = fill_sym;
                }
            }
            block_syms[block_len..padded_len].fill(fill_sym);
            bit_writer.finish();
        }
        Mask::Values(mv) => {
            let bits = mv.bit_buffer();
            for (i, &v) in block_vals.iter().enumerate() {
                block_syms[i] = if bits.value(block_start + i) {
                    map_int_to_u64(v, is_signed, signed_zigzag) as u8
                } else {
                    fill_sym
                };
            }
            block_syms[block_len..padded_len].fill(fill_sym);
        }
    }
}

/// Validates that all `num_blocks` headers in `headers_bytes` lie cleanly within `payload_len`
/// relative to `payload_base_offset`.
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
    for b in 0..num_blocks {
        let hdr = RansBlockHeader::read_at(headers_bytes, b);
        let rans_start = usize::try_from(hdr.rans_byte_offset)?
            .checked_sub(base)
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    "Rans block {b} rans_byte_offset {} precedes payload_base_offset {base}",
                    hdr.rans_byte_offset
                )
            })?;
        let rans_bytes_len = BLOCK_STATES_BYTES
            .checked_add((hdr.rans_u16_len as usize).saturating_mul(2))
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} rans byte length overflow"))?;
        let rans_end = rans_start
            .checked_add(rans_bytes_len)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} rans end offset overflow"))?;
        vortex_ensure!(
            rans_end <= payload_len,
            "Rans block {b} rans slice {rans_start}..{rans_end} exceeds payload length {payload_len}"
        );

        let raw_start = usize::try_from(hdr.raw_byte_offset)?
            .checked_sub(base)
            .ok_or_else(|| {
                vortex_error::vortex_err!(
                    "Rans block {b} raw_byte_offset {} precedes payload_base_offset {base}",
                    hdr.raw_byte_offset
                )
            })?;
        if !has_extra_bits {
            vortex_ensure_eq!(
                hdr.raw_u64_len,
                0,
                "Rans block {b} has raw_u64_len {} when has_extra_bits is false",
                hdr.raw_u64_len
            );
        }
        let raw_bytes_len = (hdr.raw_u64_len as usize)
            .checked_mul(8)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} raw byte length overflow"))?;
        let raw_end = raw_start
            .checked_add(raw_bytes_len)
            .ok_or_else(|| vortex_error::vortex_err!("Rans block {b} raw end offset overflow"))?;
        vortex_ensure!(
            rans_end <= raw_start && raw_end <= payload_len,
            "Rans block {b} raw slice {raw_start}..{raw_end} overlaps rans_end {rans_end} or exceeds payload length {payload_len}"
        );
    }
    Ok(())
}

/// Decodes the first `n_stripes` (`1..=128`) of `block_idx` into `dst[..n_stripes * 32]`.
#[expect(clippy::too_many_arguments)]
#[inline]
pub fn decode_block_stripes<T>(
    table: &RansDecodeTable,
    headers_bytes: &[u8],
    payload_bytes: &[u8],
    payload_base_offset: u32,
    block_idx: usize,
    n_stripes: usize,
    signed_zigzag: bool,
    has_extra_bits: bool,
    dst: &mut [T],
) where
    T: NativePType,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
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
    let words_end = words_start + (hdr.rans_u16_len as usize) * 2;
    let word_chunks = payload_bytes[words_start..words_end].as_chunks::<2>().0;

    if !has_extra_bits {
        decode_stripes_no_extra::<T>(
            table,
            &mut states,
            word_chunks,
            n_stripes,
            signed_zigzag,
            dst,
        );
    } else {
        let raw_start = (hdr.raw_byte_offset as usize) - base;
        let raw_end = raw_start + (hdr.raw_u64_len as usize) * 8;
        let raw_bytes = &payload_bytes[raw_start..raw_end];
        decode_stripes_with_extra::<T>(
            table,
            &mut states,
            word_chunks,
            raw_bytes,
            n_stripes,
            signed_zigzag,
            dst,
        );
    }
}

/// Fast path when `has_extra_bits == false`: pure 32-lane state update + branchless `u16` refill,
/// completely skipping `BASE_TABLE` and the `raw_bits` stream (`sym` IS the exact unsigned value!).
fn decode_stripes_no_extra<T>(
    table: &RansDecodeTable,
    states: &mut [u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    n_stripes: usize,
    signed_zigzag: bool,
    dst: &mut [T],
) where
    T: NativePType,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let mut word_pos = 0usize;
    let mut syms = [0u8; NUM_LANES];
    let mut next_states = [0u32; NUM_LANES];

    for stripe in 0..n_stripes {
        // Phase A: 32 independent L1 table lookups and multiply-adds.
        for lane in 0..NUM_LANES {
            let state = states[lane];
            let slot = (state & PROB_MASK) as usize;
            let entry = table.slots[slot];
            syms[lane] = entry.sym;
            next_states[lane] =
                u32::from(entry.freq) * (state >> PROB_BITS) + u32::from(entry.bias);
        }

        // Phase B: single-refill renormalization (100% branchless `csel` + `cinc` when 32 words
        // of slack remain in `word_chunks`).
        if word_pos + NUM_LANES <= word_chunks.len() {
            for lane in 0..NUM_LANES {
                let s = next_states[lane];
                let need = s < RANS_L;
                // SAFETY: `word_pos` starts at `<= word_chunks.len() - 32` and increments at most
                // once per lane across `0..32`, so `word_pos < word_chunks.len()` holds for every lane.
                let word = u32::from(u16::from_le_bytes(*unsafe {
                    word_chunks.get_unchecked(word_pos)
                }));
                states[lane] = if need { (s << 16) | word } else { s };
                word_pos += usize::from(need);
            }
        } else {
            for lane in 0..NUM_LANES {
                let mut s = next_states[lane];
                if s < RANS_L && word_pos < word_chunks.len() {
                    let word = u32::from(u16::from_le_bytes(word_chunks[word_pos]));
                    word_pos += 1;
                    s = (s << 16) | word;
                }
                states[lane] = s;
            }
        }

        // Phase C: SIMD-friendly 32-wide output conversion.
        let out_stripe = &mut dst[stripe * NUM_LANES..(stripe + 1) * NUM_LANES];
        if signed_zigzag {
            for lane in 0..NUM_LANES {
                let u = u64::from(syms[lane]);
                out_stripe[lane] = zigzag_decode(u).as_();
            }
        } else {
            for lane in 0..NUM_LANES {
                out_stripe[lane] = u64::from(syms[lane]).as_();
            }
        }
    }
}

/// General path when `has_extra_bits == true`: 32-lane state update + branchless `u16` refill +
/// forward `u128` raw-bits reader.
fn decode_stripes_with_extra<T>(
    table: &RansDecodeTable,
    states: &mut [u32; NUM_LANES],
    word_chunks: &[[u8; 2]],
    raw_bytes: &[u8],
    n_stripes: usize,
    signed_zigzag: bool,
    dst: &mut [T],
) where
    T: NativePType,
    i64: AsPrimitive<T>,
    u64: AsPrimitive<T>,
{
    let mut word_pos = 0usize;
    let mut bit_reader = BitReader::new(raw_bytes);
    let mut syms = [0u8; NUM_LANES];
    let mut ebits_arr = [0u8; NUM_LANES];
    let mut next_states = [0u32; NUM_LANES];

    for stripe in 0..n_stripes {
        // Phase A: 32 independent L1 table lookups and multiply-adds.
        for lane in 0..NUM_LANES {
            let state = states[lane];
            let slot = (state & PROB_MASK) as usize;
            let entry = table.slots[slot];
            syms[lane] = entry.sym;
            ebits_arr[lane] = entry.ebits;
            next_states[lane] =
                u32::from(entry.freq) * (state >> PROB_BITS) + u32::from(entry.bias);
        }

        // Phase B: branchless `u16` refill whenever 32 words remain.
        if word_pos + NUM_LANES <= word_chunks.len() {
            for lane in 0..NUM_LANES {
                let s = next_states[lane];
                let need = s < RANS_L;
                // SAFETY: `word_pos_0 + 32 <= word_chunks.len()` and `word_pos` advances by `<= 1`
                // per lane, so every `word_pos` is strictly `< word_chunks.len()`.
                let word = u32::from(u16::from_le_bytes(*unsafe {
                    word_chunks.get_unchecked(word_pos)
                }));
                states[lane] = if need { (s << 16) | word } else { s };
                word_pos += usize::from(need);
            }
        } else {
            for lane in 0..NUM_LANES {
                let mut s = next_states[lane];
                if s < RANS_L && word_pos < word_chunks.len() {
                    let word = u32::from(u16::from_le_bytes(word_chunks[word_pos]));
                    word_pos += 1;
                    s = (s << 16) | word;
                }
                states[lane] = s;
            }
        }

        // Phase C: reconstruct `u = BASE_TABLE[sym] | raw` from forward `u128` bit-reader.
        let out_stripe = &mut dst[stripe * NUM_LANES..(stripe + 1) * NUM_LANES];
        if signed_zigzag {
            for lane in 0..NUM_LANES {
                let base = BASE_TABLE[usize::from(syms[lane])];
                let raw = bit_reader.read(ebits_arr[lane]);
                out_stripe[lane] = zigzag_decode(base | raw).as_();
            }
        } else {
            for lane in 0..NUM_LANES {
                let base = BASE_TABLE[usize::from(syms[lane])];
                let raw = bit_reader.read(ebits_arr[lane]);
                out_stripe[lane] = (base | raw).as_();
            }
        }
    }
}
