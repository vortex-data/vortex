// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Windowed rank/select over the upper bit array, taking a half-open bit range so a sample can give
//! the scan its lower bound.
//!
//! Reads the backing bytes directly, so a lookup allocates nothing. [`LOG_SAMPLING0`] and
//! [`LOG_SAMPLING1`] cap a window at 512 unset or 256 set bits — eight to sixteen words — which a
//! scalar walk covers without vectorising.
//!
//! [`LOG_SAMPLING0`]: super::LOG_SAMPLING0
//! [`LOG_SAMPLING1`]: super::LOG_SAMPLING1

/// A window of bits: the backing bytes, the bit offset of bit zero within them, and a length.
#[derive(Clone, Copy, Debug)]
pub struct Bits<'a> {
    bytes: &'a [u8],
    offset: usize,
    len: usize,
}

impl<'a> Bits<'a> {
    /// A window of `len` bits beginning `offset` bits into `bytes`.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is too short to hold `offset + len` bits.
    #[inline]
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(
            offset.saturating_add(len) <= bytes.len().saturating_mul(8),
            "{} bytes cannot hold {len} bits at offset {offset}",
            bytes.len()
        );
        Self { bytes, offset, len }
    }

    /// The number of bits in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the window holds no bits at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Returns the position of the `nth` set bit within `[start, end)` of `bits`, relative to `start`,
/// or `None` if the range holds `nth` set bits or fewer.
///
/// # Panics
///
/// Panics if `start > end` or `end > bits.len()`.
#[inline]
pub fn select_range(bits: Bits<'_>, start: usize, end: usize, nth: usize) -> Option<usize> {
    select_in_range(bits, start, end, nth, false)
}

/// Returns the position of the `nth` unset bit within `[start, end)` of `bits`, relative to
/// `start`, or `None` if the range holds `nth` unset bits or fewer.
///
/// The complement of [`select_range`]: a bucket boundary in the upper array is a zero, so this is
/// what turns a high part into a rank.
///
/// # Panics
///
/// Panics if `start > end` or `end > bits.len()`.
#[inline]
pub fn select_zero_range(bits: Bits<'_>, start: usize, end: usize, nth: usize) -> Option<usize> {
    select_in_range(bits, start, end, nth, true)
}

/// The bits of `[start, end)` as whole `u64` words at a zero bit offset, so a decode shifts
/// once here rather than per word.
///
/// The last word is zero-padded above the window's end, so counting set bits across the result
/// counts exactly the window's. At roughly two bits per element the allocation is `n / 32` words —
/// a few KB per million rows.
///
/// # Panics
///
/// Panics if `start > end` or `end > bits.len()`.
pub fn window_words(bits: Bits<'_>, start: usize, end: usize) -> Vec<u64> {
    assert!(start <= end, "start {start} exceeds end {end}");
    assert!(end <= bits.len, "end {end} exceeds len {}", bits.len);

    let total = end - start;
    let base = bits.offset + start;
    let mut words = Vec::with_capacity(total.div_ceil(u64::BITS as usize));
    let mut pos = 0usize;
    while pos < total {
        let width = (total - pos).min(64);
        words.push(load_bits(bits.bytes, base + pos, width));
        pos += width;
    }
    words
}

/// Shared walk behind [`select_range`] (`zeros == false`) and [`select_zero_range`]
/// (`zeros == true`).
#[inline]
fn select_in_range(
    bits: Bits<'_>,
    start: usize,
    end: usize,
    nth: usize,
    zeros: bool,
) -> Option<usize> {
    assert!(start <= end, "start {start} exceeds end {end}");
    assert!(end <= bits.len, "end {end} exceeds len {}", bits.len);

    // The window begins `offset + start` bits into the backing bytes.
    let bytes = bits.bytes;
    let base = bits.offset + start;
    let total = end - start;

    let mut remaining = nth;
    let mut pos = 0usize;
    while pos < total {
        let width = (total - pos).min(64);
        let mut word = load_bits(bytes, base + pos, width);
        if zeros {
            // Complementing turns the padding above `width` into ones, so mask it off again.
            word = mask_to(!word, width);
        }
        let count = word.count_ones() as usize;
        if remaining < count {
            return Some(pos + select_in_word(word, remaining));
        }
        remaining -= count;
        pos += width;
    }
    None
}

/// Reads `width` bits starting at absolute bit `at`, returned in the low bits with the rest zero.
///
/// An unaligned 64 bits straddles nine bytes: the common path takes the first eight as one
/// little-endian word and folds the ninth in across the shift. Within nine bytes of the end the
/// slower path assembles whatever is there, since the bytes it cannot read land above `width`.
#[inline]
fn load_bits(bytes: &[u8], at: usize, width: usize) -> u64 {
    debug_assert!(width > 0 && width <= 64, "width {width} out of range");

    let first = at / 8;
    let shift = at % 8;

    let head = bytes.get(first..).and_then(<[u8]>::first_chunk::<8>);
    let word = match (head, bytes.get(first + 8)) {
        (Some(chunk), Some(&straddle)) => {
            let lo = u64::from_le_bytes(*chunk);
            // Shifting a `u64` by 64 is undefined, so the aligned case needs its own arm.
            if shift == 0 {
                lo
            } else {
                (lo >> shift) | (u64::from(straddle) << (64 - shift))
            }
        }
        _ => {
            let mut raw = 0u128;
            for (i, &byte) in bytes.iter().skip(first).take(9).enumerate() {
                raw |= u128::from(byte) << (8 * i);
            }
            (raw >> shift) as u64
        }
    };
    mask_to(word, width)
}

/// Clears every bit at or above `width`.
#[inline]
fn mask_to(word: u64, width: usize) -> u64 {
    if width == 64 {
        word
    } else {
        word & ((1u64 << width) - 1)
    }
}

/// Returns the index of the `nth` set bit of `word`, which must hold more than `nth` set bits.
///
/// Narrows a byte at a time first, so [`select_in_byte`] loops at most seven times.
#[inline]
fn select_in_word(word: u64, nth: usize) -> usize {
    debug_assert!(
        nth < word.count_ones() as usize,
        "rank {nth} is not present in the word"
    );

    let mut remaining = nth;
    let mut rest = word;
    let mut shift = 0usize;
    loop {
        let byte = (rest & 0xFF) as u8;
        let count = byte.count_ones() as usize;
        if remaining < count {
            return shift + select_in_byte(byte, remaining);
        }
        remaining -= count;
        rest >>= 8;
        shift += 8;
    }
}

/// Returns the index of the `nth` set bit of `byte`.
///
/// `byte & (byte - 1)` clears the lowest set bit, so `nth` of those leave the target lowest.
#[inline]
fn select_in_byte(byte: u8, nth: usize) -> usize {
    let mut rest = byte;
    for _ in 0..nth {
        rest &= rest - 1;
    }
    rest.trailing_zeros() as usize
}
