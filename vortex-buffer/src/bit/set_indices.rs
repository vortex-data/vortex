// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast extraction of set-bit indices from packed bitmaps.
//!
//! Provides both an iterator-based API and a bulk collection API, with
//! scalar and (on x86-64) SIMD-accelerated implementations.

use crate::bit::count_ones::count_ones;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Mask a u64 to keep only bits `[lo .. lo + count)`.
#[inline(always)]
fn mask_word(word: u64, lo: usize, count: usize) -> u64 {
    debug_assert!(lo + count <= 64);
    if count == 0 {
        return 0;
    }
    let shifted = word >> lo;
    if count >= 64 {
        shifted
    } else {
        shifted & ((1u64 << count) - 1)
    }
}

/// Read a little-endian u64 from a pointer to at least 8 bytes.
///
/// # Safety
/// `ptr` must be valid for reads of 8 bytes.
#[inline(always)]
unsafe fn read_u64_le(ptr: *const u8) -> u64 {
    // Unaligned read — always valid on x86, and the compiler does the right
    // thing on other architectures.
    unsafe { (ptr as *const u64).read_unaligned().to_le() }
}

/// Load up to 8 bytes from a raw pointer into a little-endian u64.
///
/// # Safety
/// `ptr` must be valid for reads of `avail` bytes.
#[inline]
unsafe fn load_partial_u64(ptr: *const u8, avail: usize) -> u64 {
    debug_assert!(avail < 8);
    let mut buf = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), avail) };
    u64::from_le_bytes(buf)
}

// ---------------------------------------------------------------------------
// Optimised scalar iterator (no SIMD)
// ---------------------------------------------------------------------------

/// A fast iterator over the indices of set bits in a packed byte buffer.
///
/// Compared to Arrow's `BitIndexIterator` this avoids the `UnalignedBitChunk`
/// abstraction and `i64` arithmetic, operating directly on `u64` words read
/// from the byte slice via raw pointer arithmetic (no bounds checks in the
/// hot loop).
pub struct ScalarBitIndexIterator<'a> {
    /// Pointer to the next group of 8 bytes.
    ptr: *const u8,
    /// End pointer for bounds.
    end: *const u8,
    /// Current u64 word being drained of set bits.
    current_word: u64,
    /// Logical bit-index base for bit 0 of the current word.
    base: usize,
    /// Logical bit-index where the next word starts.
    next_word_base: usize,
    /// Total number of logical bits remaining after the current word.
    remaining: usize,
    /// Phantom lifetime.
    _marker: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> ScalarBitIndexIterator<'a> {
    /// Create a new iterator over set-bit indices.
    ///
    /// `buffer` is the packed byte slice, `offset` is the starting bit offset,
    /// and `len` is the number of bits to scan.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: buffer.as_ptr(),
                end: buffer.as_ptr(),
                current_word: 0,
                base: 0,
                next_word_base: 0,
                remaining: 0,
                _marker: std::marker::PhantomData,
            };
        }

        let start_byte = offset / 8;
        let start_bit = offset % 8;
        let bytes = &buffer[start_byte..];
        let first_word_bits = (64 - start_bit).min(len);

        // SAFETY: `bytes` is a valid slice; we use pointer arithmetic
        // within its bounds.
        let (first_word, ptr, end) = if bytes.len() >= 8 {
            let word = unsafe { read_u64_le(bytes.as_ptr()) };
            let masked = mask_word(word, start_bit, first_word_bits);
            let ptr = unsafe { bytes.as_ptr().add(8) };
            let end = unsafe { bytes.as_ptr().add(bytes.len()) };
            (masked, ptr, end)
        } else {
            let word = unsafe { load_partial_u64(bytes.as_ptr(), bytes.len()) };
            let masked = mask_word(word, start_bit, first_word_bits);
            let ptr = bytes.as_ptr();
            let end = bytes.as_ptr();
            (masked, ptr, end)
        };

        Self {
            ptr,
            end,
            current_word: first_word,
            base: 0,
            next_word_base: first_word_bits,
            remaining: len - first_word_bits,
            _marker: std::marker::PhantomData,
        }
    }
}

impl Iterator for ScalarBitIndexIterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let tz = self.current_word.trailing_zeros() as usize;
                // Clear lowest set bit: x & (x - 1)  (BLSR on BMI1)
                self.current_word &= self.current_word - 1;
                return Some(self.base + tz);
            }

            if self.remaining == 0 {
                return None;
            }

            self.base = self.next_word_base;
            let bits_this_word = self.remaining.min(64);
            self.next_word_base = self.base + bits_this_word;

            // SAFETY: ptr/end stay within the original buffer slice.
            let avail = unsafe { self.end.offset_from(self.ptr) } as usize;
            if avail >= 8 {
                self.current_word = unsafe { read_u64_le(self.ptr) };
                self.ptr = unsafe { self.ptr.add(8) };
            } else {
                self.current_word = unsafe { load_partial_u64(self.ptr, avail) };
                self.ptr = self.end;
            }

            if bits_this_word < 64 {
                self.current_word &= (1u64 << bits_this_word) - 1;
            }

            self.remaining -= bits_this_word;
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining + 64))
    }
}

// ---------------------------------------------------------------------------
// Bulk collection: scalar
// ---------------------------------------------------------------------------

/// Collect all set-bit indices into a `Vec<u32>`. Faster than iterating
/// one-by-one because it pre-allocates and avoids per-element iterator overhead.
///
/// Uses `u32` indices to halve memory bandwidth (sufficient for buffers up to
/// 4 billion bits).
#[allow(clippy::cast_possible_truncation)]
pub fn collect_set_indices_scalar(buffer: &[u8], offset: usize, len: usize) -> Vec<u32> {
    if len == 0 {
        return Vec::new();
    }

    let count = count_ones(buffer, offset, len);
    let mut out = Vec::with_capacity(count);

    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let bytes = &buffer[start_byte..];
    let first_word_bits = (64 - start_bit).min(len);

    let mut remaining = len - first_word_bits;
    let mut base = 0u32;
    let mut ptr = bytes.as_ptr();
    let end = unsafe { bytes.as_ptr().add(bytes.len()) };

    // First (possibly partial) word.
    let first_word = if bytes.len() >= 8 {
        let word = unsafe { read_u64_le(ptr) };
        ptr = unsafe { ptr.add(8) };
        mask_word(word, start_bit, first_word_bits)
    } else {
        mask_word(
            unsafe { load_partial_u64(ptr, bytes.len()) },
            start_bit,
            first_word_bits,
        )
    };
    drain_word_to_vec(first_word, base, &mut out);
    // first_word_bits <= 64, so fits in u32
    base = first_word_bits as u32;

    // Full u64 words.
    while remaining >= 64 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        if avail < 8 {
            break;
        }
        let word = unsafe { read_u64_le(ptr) };
        drain_word_to_vec(word, base, &mut out);
        ptr = unsafe { ptr.add(8) };
        base += 64;
        remaining -= 64;
    }

    // Final partial word.
    if remaining > 0 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        let word = unsafe { load_partial_u64(ptr, avail) };
        let masked = word & ((1u64 << remaining) - 1);
        drain_word_to_vec(masked, base, &mut out);
    }

    out
}

/// Extract all set-bit positions from a u64 and append them to `out`.
#[inline]
fn drain_word_to_vec(word: u64, base: u32, out: &mut Vec<u32>) {
    if word == u64::MAX {
        out.reserve(64);
        for i in 0..64u32 {
            out.push(base + i);
        }
        return;
    }
    let mut w = word;
    while w != 0 {
        let tz = w.trailing_zeros();
        out.push(base + tz);
        w &= w - 1; // BLSR
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated bulk collection (x86-64 only)
// ---------------------------------------------------------------------------

/// Collect set-bit indices using the best available method for this platform.
///
/// Uses BMI2 hardware instructions on x86-64 when available for faster
/// bit extraction. Falls back to scalar implementation otherwise.
pub fn collect_set_indices(buffer: &[u8], offset: usize, len: usize) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            // SAFETY: feature detection guarantees BMI2.
            return unsafe { collect_set_indices_bmi2(buffer, offset, len) };
        }
    }

    collect_set_indices_scalar(buffer, offset, len)
}

// ---------------------------------------------------------------------------
// BMI2 implementation — uses BLSR and TZCNT hardware instructions
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,bmi2")]
#[allow(clippy::cast_possible_truncation)]
unsafe fn collect_set_indices_bmi2(buffer: &[u8], offset: usize, len: usize) -> Vec<u32> {
    if len == 0 {
        return Vec::new();
    }

    let count = count_ones(buffer, offset, len);
    let mut out: Vec<u32> = Vec::with_capacity(count);
    let mut dst = out.as_mut_ptr();

    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let bytes = &buffer[start_byte..];
    let first_word_bits = (64 - start_bit).min(len);

    let mut remaining = len - first_word_bits;
    let mut base = 0u32;
    let mut ptr = bytes.as_ptr();
    let end = unsafe { bytes.as_ptr().add(bytes.len()) };

    // First (possibly partial) word.
    let first_word = if bytes.len() >= 8 {
        let word = unsafe { read_u64_le(ptr) };
        ptr = unsafe { ptr.add(8) };
        mask_word(word, start_bit, first_word_bits)
    } else {
        mask_word(
            unsafe { load_partial_u64(ptr, bytes.len()) },
            start_bit,
            first_word_bits,
        )
    };
    dst = unsafe { drain_word_to_ptr(first_word, base, dst) };
    // first_word_bits <= 64, fits in u32
    base = first_word_bits as u32;

    // Full u64 words.
    while remaining >= 64 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        if avail < 8 {
            break;
        }
        let word = unsafe { read_u64_le(ptr) };
        dst = unsafe { drain_word_to_ptr(word, base, dst) };
        ptr = unsafe { ptr.add(8) };
        base += 64;
        remaining -= 64;
    }

    // Final partial word.
    if remaining > 0 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        let word = unsafe { load_partial_u64(ptr, avail) };
        let masked = if remaining < 64 {
            word & ((1u64 << remaining) - 1)
        } else {
            word
        };
        dst = unsafe { drain_word_to_ptr(masked, base, dst) };
    }

    // SAFETY: we wrote exactly `count` u32 values through `dst`.
    let written = unsafe { dst.offset_from(out.as_ptr()) } as usize;
    debug_assert_eq!(written, count);
    unsafe { out.set_len(written) };

    out
}

/// Drain set bits from `word` into the raw pointer `dst`, returning the
/// advanced pointer.
///
/// For fully-set words (common at high density), writes 64 sequential
/// indices in a tight loop without any bit manipulation.
///
/// # Safety
/// Caller must ensure `dst` has room for `word.count_ones()` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "bmi1,bmi2")]
unsafe fn drain_word_to_ptr(word: u64, base: u32, mut dst: *mut u32) -> *mut u32 {
    if word == u64::MAX {
        // Fast path for fully-set words (common at high density).
        unsafe {
            for i in 0..64u32 {
                dst.add(i as usize).write(base + i);
            }
            return dst.add(64);
        }
    }

    let mut w = word;
    while w != 0 {
        let tz = w.trailing_zeros();
        unsafe {
            dst.write(base + tz);
            dst = dst.add(1);
            w = core::arch::x86_64::_blsr_u64(w);
        }
    }
    dst
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::BitBuffer;

    fn arrow_set_indices(buf: &BitBuffer) -> Vec<usize> {
        use arrow_buffer::bit_iterator::BitIndexIterator;
        BitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect()
    }

    #[rstest]
    #[case(128)]
    #[case(1024)]
    #[case(2048)]
    #[case(16384)]
    #[case(65536)]
    fn test_scalar_iterator_matches_arrow(#[case] len: usize) {
        let buf = BitBuffer::from_iter((0..len).map(|i| i % 2 == 0));
        let expected = arrow_set_indices(&buf);
        let actual: Vec<usize> =
            ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect();
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(128)]
    #[case(1024)]
    #[case(2048)]
    #[case(16384)]
    #[case(65536)]
    fn test_collect_scalar_matches_arrow(#[case] len: usize) {
        let buf = BitBuffer::from_iter((0..len).map(|i| i % 2 == 0));
        let expected: Vec<u32> = arrow_set_indices(&buf).iter().map(|&i| i as u32).collect();
        let actual = collect_set_indices_scalar(buf.inner().as_slice(), buf.offset(), buf.len());
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(128)]
    #[case(1024)]
    #[case(2048)]
    #[case(16384)]
    #[case(65536)]
    fn test_collect_simd_matches_arrow(#[case] len: usize) {
        let buf = BitBuffer::from_iter((0..len).map(|i| i % 2 == 0));
        let expected: Vec<u32> = arrow_set_indices(&buf).iter().map(|&i| i as u32).collect();
        let actual = collect_set_indices(buf.inner().as_slice(), buf.offset(), buf.len());
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(7)]
    #[case(8)]
    #[case(63)]
    #[case(64)]
    #[case(65)]
    #[case(127)]
    #[case(128)]
    #[case(129)]
    fn test_scalar_various_sizes(#[case] len: usize) {
        let buf = BitBuffer::from_iter((0..len).map(|i| i % 2 == 0));
        let expected = arrow_set_indices(&buf);
        let actual: Vec<usize> =
            ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect();
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(1)]
    #[case(3)]
    #[case(5)]
    #[case(7)]
    fn test_with_offset(#[case] offset: usize) {
        let total = 128;
        let buf = BitBuffer::from_iter((0..total).map(|i| i % 3 == 0));
        let sliced = buf.slice(offset..total);
        let expected = arrow_set_indices(&sliced);
        let actual: Vec<usize> =
            ScalarBitIndexIterator::new(sliced.inner().as_slice(), sliced.offset(), sliced.len())
                .collect();
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(1)]
    #[case(3)]
    #[case(5)]
    #[case(7)]
    fn test_collect_with_offset(#[case] offset: usize) {
        let total = 128;
        let buf = BitBuffer::from_iter((0..total).map(|i| i % 3 == 0));
        let sliced = buf.slice(offset..total);
        let expected: Vec<u32> = arrow_set_indices(&sliced)
            .iter()
            .map(|&i| i as u32)
            .collect();
        let actual_scalar =
            collect_set_indices_scalar(sliced.inner().as_slice(), sliced.offset(), sliced.len());
        assert_eq!(expected, actual_scalar);
        let actual_simd =
            collect_set_indices(sliced.inner().as_slice(), sliced.offset(), sliced.len());
        assert_eq!(expected, actual_simd);
    }

    #[test]
    fn test_dense_pattern() {
        let buf = BitBuffer::new_set(256);
        let expected = arrow_set_indices(&buf);
        let actual: Vec<usize> =
            ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_sparse_pattern() {
        let buf = BitBuffer::from_iter((0..1024).map(|i| i == 7 || i == 500 || i == 1023));
        let expected = arrow_set_indices(&buf);
        let actual: Vec<usize> =
            ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_collect_dense() {
        let buf = BitBuffer::new_set(256);
        let expected: Vec<u32> = (0..256u32).collect();
        let actual_scalar =
            collect_set_indices_scalar(buf.inner().as_slice(), buf.offset(), buf.len());
        assert_eq!(expected, actual_scalar);
        let actual_simd = collect_set_indices(buf.inner().as_slice(), buf.offset(), buf.len());
        assert_eq!(expected, actual_simd);
    }
}
