// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast extraction of set-bit indices from packed bitmaps.
//!
//! Provides both an iterator-based API and a bulk collection API, with
//! scalar and (on x86-64) SIMD-accelerated implementations.
//!
//! The fastest path uses AVX-512 VPCOMPRESSD to convert 16 bitmap bits into
//! compressed index stores in a single instruction, combined with 512-bit
//! zero-checks to skip sparse regions. Falls back to BMI2 BLSR/TZCNT, then
//! pure scalar.

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
    unsafe { (ptr as *const u64).read_unaligned().to_le() }
}

/// Load up to 7 bytes from a raw pointer into a little-endian u64.
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

/// Load the first (possibly partial) u64 word from a byte buffer, masking off
/// bits before `start_bit`. Advances `ptr` past consumed bytes.
///
/// # Safety
/// `ptr` must be valid, `end - ptr >= 1`.
#[inline]
unsafe fn load_first_word(
    ptr: &mut *const u8,
    end: *const u8,
    start_bit: usize,
    len: usize,
) -> (u64, usize) {
    let avail = unsafe { end.offset_from(*ptr) } as usize;
    let first_word_bits = (64 - start_bit).min(len);
    let word = if avail >= 8 {
        let w = unsafe { read_u64_le(*ptr) };
        *ptr = unsafe { (*ptr).add(8) };
        w
    } else {
        unsafe { load_partial_u64(*ptr, avail) }
    };
    (mask_word(word, start_bit, first_word_bits), first_word_bits)
}

// ---------------------------------------------------------------------------
// Optimised scalar iterator — skips zero words fast
// ---------------------------------------------------------------------------

/// A fast iterator over the indices of set bits in a packed byte buffer.
///
/// Optimised for sparse bitmaps: scans multiple u64 words at once to skip
/// over zero regions cheaply. At low density (≤20%) this significantly
/// reduces branch overhead compared to checking one word at a time.
pub struct ScalarBitIndexIterator<'a> {
    /// Pointer to the next u64 word.
    ptr: *const u8,
    /// End pointer (one past last valid byte).
    end: *const u8,
    /// Current u64 word being drained of set bits.
    current_word: u64,
    /// Logical bit-index base for bit 0 of the current word.
    base: usize,
    /// Logical bit-index where the next word starts.
    next_base: usize,
    /// Bits remaining after current word.
    remaining: usize,
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
                next_base: 0,
                remaining: 0,
                _marker: std::marker::PhantomData,
            };
        }

        let start_byte = offset / 8;
        let start_bit = offset % 8;
        let bytes = &buffer[start_byte..];
        let mut ptr = bytes.as_ptr();
        let end = unsafe { bytes.as_ptr().add(bytes.len()) };

        // SAFETY: bytes is valid
        let (first_word, first_bits) = unsafe { load_first_word(&mut ptr, end, start_bit, len) };

        Self {
            ptr,
            end,
            current_word: first_word,
            base: 0,
            next_base: first_bits,
            remaining: len - first_bits,
            _marker: std::marker::PhantomData,
        }
    }

    /// Advance past zero words in batches. Returns the next non-zero word
    /// (masked if final), updating base/remaining, or None if exhausted.
    #[inline]
    fn advance_to_nonzero(&mut self) -> bool {
        let avail = unsafe { self.end.offset_from(self.ptr) } as usize;

        // Fast path: skip 4 zero words at a time (256 bits).
        if self.remaining >= 256 && avail >= 32 {
            loop {
                let w0 = unsafe { read_u64_le(self.ptr) };
                let w1 = unsafe { read_u64_le(self.ptr.add(8)) };
                let w2 = unsafe { read_u64_le(self.ptr.add(16)) };
                let w3 = unsafe { read_u64_le(self.ptr.add(24)) };

                if (w0 | w1 | w2 | w3) != 0 {
                    self.base = self.next_base;
                    if w0 != 0 {
                        self.current_word = w0;
                        self.ptr = unsafe { self.ptr.add(8) };
                        self.next_base = self.base + 64;
                        self.remaining -= 64;
                        return true;
                    }
                    self.base += 64;
                    if w1 != 0 {
                        self.current_word = w1;
                        self.ptr = unsafe { self.ptr.add(16) };
                        self.next_base = self.base + 64;
                        self.remaining -= 128;
                        return true;
                    }
                    self.base += 64;
                    if w2 != 0 {
                        self.current_word = w2;
                        self.ptr = unsafe { self.ptr.add(24) };
                        self.next_base = self.base + 64;
                        self.remaining -= 192;
                        return true;
                    }
                    self.base += 64;
                    self.current_word = w3;
                    self.ptr = unsafe { self.ptr.add(32) };
                    self.next_base = self.base + 64;
                    self.remaining -= 256;
                    return true;
                }

                self.ptr = unsafe { self.ptr.add(32) };
                self.next_base += 256;
                self.remaining -= 256;

                let new_avail = unsafe { self.end.offset_from(self.ptr) } as usize;
                if self.remaining < 256 || new_avail < 32 {
                    break;
                }
            }
        }

        // Tail: one word at a time.
        while self.remaining > 0 {
            self.base = self.next_base;
            let bits = self.remaining.min(64);
            self.next_base = self.base + bits;

            let word_avail = unsafe { self.end.offset_from(self.ptr) } as usize;
            let mut word = if word_avail >= 8 {
                let w = unsafe { read_u64_le(self.ptr) };
                self.ptr = unsafe { self.ptr.add(8) };
                w
            } else {
                let w = unsafe { load_partial_u64(self.ptr, word_avail) };
                self.ptr = self.end;
                w
            };

            if bits < 64 {
                word &= (1u64 << bits) - 1;
            }
            self.remaining -= bits;

            if word != 0 {
                self.current_word = word;
                return true;
            }
        }

        false
    }
}

impl Iterator for ScalarBitIndexIterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        // Hot path: drain bits from current word.
        if self.current_word != 0 {
            let tz = self.current_word.trailing_zeros() as usize;
            self.current_word &= self.current_word - 1; // BLSR
            return Some(self.base + tz);
        }

        // Advance to next non-zero word (skipping zero words in bulk).
        self.advance_to_nonzero().then(|| {
            let tz = self.current_word.trailing_zeros() as usize;
            self.current_word &= self.current_word - 1;
            self.base + tz
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining + 64))
    }
}

// ---------------------------------------------------------------------------
// Bulk collection: scalar
// ---------------------------------------------------------------------------

/// Collect all set-bit indices into a `Vec<u32>`. Scalar fallback that works
/// on all platforms.
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
    let mut ptr = bytes.as_ptr();
    let end = unsafe { bytes.as_ptr().add(bytes.len()) };

    let (first_word, first_bits) = unsafe { load_first_word(&mut ptr, end, start_bit, len) };
    drain_word_to_vec(first_word, 0, &mut out);
    let mut base = first_bits as u32;
    let mut remaining = len - first_bits;

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

    if remaining > 0 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        let word = unsafe { load_partial_u64(ptr, avail) };
        let masked = word & ((1u64 << remaining) - 1);
        drain_word_to_vec(masked, base, &mut out);
    }

    out
}

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
        w &= w - 1;
    }
}

// ---------------------------------------------------------------------------
// Dispatch: pick the best available SIMD path
// ---------------------------------------------------------------------------

/// Collect set-bit indices using the best available SIMD path.
///
/// If the true count is already known (e.g. from a validity buffer's cached
/// null count), pass it via `true_count` to skip the `count_ones` pre-pass.
/// This eliminates a full scan of the bitmap and makes the function
/// competitive with the iterator even at very low densities.
pub fn collect_set_indices(buffer: &[u8], offset: usize, len: usize) -> Vec<u32> {
    collect_set_indices_with_count(buffer, offset, len, None)
}

/// Like [`collect_set_indices`], but with a pre-known true count.
pub fn collect_set_indices_with_count(
    buffer: &[u8],
    offset: usize,
    len: usize,
    true_count: Option<usize>,
) -> Vec<u32> {
    if len == 0 {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        let has_avx512 = is_x86_feature_detected!("avx512f");
        let has_bmi2 = is_x86_feature_detected!("bmi2");

        if has_avx512 || has_bmi2 {
            let count = true_count.unwrap_or_else(|| count_ones(buffer, offset, len));

            if has_avx512 && count * 8 > len {
                return unsafe { collect_set_indices_avx512(buffer, offset, len, count) };
            }
            if has_bmi2 {
                return unsafe { collect_set_indices_bmi2(buffer, offset, len, count) };
            }
        }
    }

    collect_set_indices_scalar(buffer, offset, len)
}

// ---------------------------------------------------------------------------
// AVX-512 implementation — VPCOMPRESSD + 512-bit zero-skip
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,bmi1,bmi2,popcnt")]
#[allow(clippy::cast_possible_truncation)]
unsafe fn collect_set_indices_avx512(
    buffer: &[u8],
    offset: usize,
    len: usize,
    count: usize,
) -> Vec<u32> {
    use std::arch::x86_64::*;
    let mut out: Vec<u32> = Vec::with_capacity(count);
    let mut dst = out.as_mut_ptr();

    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let bytes = &buffer[start_byte..];
    let mut ptr = bytes.as_ptr();
    let end = unsafe { bytes.as_ptr().add(bytes.len()) };

    let (first_word, first_bits) = unsafe { load_first_word(&mut ptr, end, start_bit, len) };
    dst = unsafe { drain_word_avx512(first_word, 0, dst) };
    let mut base = first_bits as u32;
    let mut remaining = len - first_bits;

    // Main loop: 8 words (512 bits) at a time.
    // _mm512_test_epi64_mask checks all 8 qwords for zero in one instruction.
    // Returns an 8-bit mask of non-zero qwords — we only process those.
    while remaining >= 512 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        if avail < 64 {
            break;
        }

        let chunk = unsafe { _mm512_loadu_si512(ptr as *const __m512i) };
        let nz_mask = _mm512_test_epi64_mask(chunk, chunk);

        if nz_mask != 0 {
            let mut m = nz_mask;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                let word = unsafe { *(ptr as *const u64).add(i) };
                let word_base = base + (i as u32 * 64);
                dst = unsafe { drain_word_avx512(word, word_base, dst) };
                m &= m - 1;
            }
        }

        ptr = unsafe { ptr.add(64) };
        base += 512;
        remaining -= 512;
    }

    // Tail: one word at a time with AVX-512 extraction.
    while remaining >= 64 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        if avail < 8 {
            break;
        }
        let word = unsafe { read_u64_le(ptr) };
        dst = unsafe { drain_word_avx512(word, base, dst) };
        ptr = unsafe { ptr.add(8) };
        base += 64;
        remaining -= 64;
    }

    if remaining > 0 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        let word = unsafe { load_partial_u64(ptr, avail) };
        let masked = if remaining < 64 {
            word & ((1u64 << remaining) - 1)
        } else {
            word
        };
        dst = unsafe { drain_word_avx512(masked, base, dst) };
    }

    let written = unsafe { dst.offset_from(out.as_ptr()) } as usize;
    debug_assert_eq!(written, count);
    unsafe { out.set_len(written) };

    out
}

/// Drain set bits from a u64 word using AVX-512 VPCOMPRESSD for dense words
/// and BMI1 BLSR for sparse words.
///
/// VPCOMPRESSD takes a 16-element vector `[base, base+1, ..., base+15]` and a
/// 16-bit mask (from the bitmap), and writes only the elements where the mask
/// bit is set, in order. This processes 16 bitmap bits per instruction.
///
/// For sparse words (≤8 set bits), the serial BLSR/TZCNT loop is faster because
/// it does ~2 cycles per bit vs VPCOMPRESSD's fixed ~16 cycles per word.
///
/// # Safety
/// Caller must ensure `dst` has room for `word.count_ones()` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,bmi1,bmi2,popcnt")]
#[inline]
unsafe fn drain_word_avx512(word: u64, base: u32, mut dst: *mut u32) -> *mut u32 {
    use std::arch::x86_64::*;

    if word == 0 {
        return dst;
    }

    let pc = word.count_ones();

    // Sparse path: BLSR loop is faster for ≤8 set bits (~2 cycles/bit vs
    // VPCOMPRESSD's fixed ~16 cycles for 4 chunks).
    if pc <= 8 {
        let mut w = word;
        while w != 0 {
            unsafe {
                dst.write(base + w.trailing_zeros());
                dst = dst.add(1);
                w = _blsr_u64(w);
            }
        }
        return dst;
    }

    // Dense path: VPCOMPRESSD processes 16 bits per instruction.
    // Split the 64-bit word into 4 × 16-bit chunks.
    let offsets = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    // Chunk 0: bits [0..16)
    let mask0 = (word & 0xFFFF) as u16;
    let base0 = _mm512_add_epi32(_mm512_set1_epi32(base as i32), offsets);
    unsafe { _mm512_mask_compressstoreu_epi32(dst as *mut i32, mask0, base0) };
    dst = unsafe { dst.add(mask0.count_ones() as usize) };

    // Chunk 1: bits [16..32)
    let mask1 = ((word >> 16) & 0xFFFF) as u16;
    let base1 = _mm512_add_epi32(_mm512_set1_epi32((base + 16) as i32), offsets);
    unsafe { _mm512_mask_compressstoreu_epi32(dst as *mut i32, mask1, base1) };
    dst = unsafe { dst.add(mask1.count_ones() as usize) };

    // Chunk 2: bits [32..48)
    let mask2 = ((word >> 32) & 0xFFFF) as u16;
    let base2 = _mm512_add_epi32(_mm512_set1_epi32((base + 32) as i32), offsets);
    unsafe { _mm512_mask_compressstoreu_epi32(dst as *mut i32, mask2, base2) };
    dst = unsafe { dst.add(mask2.count_ones() as usize) };

    // Chunk 3: bits [48..64)
    let mask3 = ((word >> 48) & 0xFFFF) as u16;
    let base3 = _mm512_add_epi32(_mm512_set1_epi32((base + 48) as i32), offsets);
    unsafe { _mm512_mask_compressstoreu_epi32(dst as *mut i32, mask3, base3) };
    dst = unsafe { dst.add(mask3.count_ones() as usize) };

    dst
}

// ---------------------------------------------------------------------------
// BMI2 implementation — BLSR/TZCNT + 4-word zero skip
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,bmi2")]
#[allow(clippy::cast_possible_truncation)]
unsafe fn collect_set_indices_bmi2(
    buffer: &[u8],
    offset: usize,
    len: usize,
    count: usize,
) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(count);
    let mut dst = out.as_mut_ptr();

    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let bytes = &buffer[start_byte..];
    let mut ptr = bytes.as_ptr();
    let end = unsafe { bytes.as_ptr().add(bytes.len()) };

    let (first_word, first_bits) = unsafe { load_first_word(&mut ptr, end, start_bit, len) };
    dst = unsafe { drain_word_bmi2(first_word, 0, dst) };
    let mut base = first_bits as u32;
    let mut remaining = len - first_bits;

    // 4-word zero skip only when sparse enough that ≥20% of groups skip.
    // At density ~0.8% (count * 128 < len), ~25% of 4-word groups are all-zero.
    // Above that, the OR + compare overhead costs more than it saves.
    if count * 128 < len {
        while remaining >= 256 {
            let avail = unsafe { end.offset_from(ptr) } as usize;
            if avail < 32 {
                break;
            }

            let w0 = unsafe { read_u64_le(ptr) };
            let w1 = unsafe { read_u64_le(ptr.add(8)) };
            let w2 = unsafe { read_u64_le(ptr.add(16)) };
            let w3 = unsafe { read_u64_le(ptr.add(24)) };

            if (w0 | w1 | w2 | w3) != 0 {
                dst = unsafe { drain_word_bmi2(w0, base, dst) };
                dst = unsafe { drain_word_bmi2(w1, base + 64, dst) };
                dst = unsafe { drain_word_bmi2(w2, base + 128, dst) };
                dst = unsafe { drain_word_bmi2(w3, base + 192, dst) };
            }

            ptr = unsafe { ptr.add(32) };
            base += 256;
            remaining -= 256;
        }
    }

    while remaining >= 64 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        if avail < 8 {
            break;
        }
        let word = unsafe { read_u64_le(ptr) };
        dst = unsafe { drain_word_bmi2(word, base, dst) };
        ptr = unsafe { ptr.add(8) };
        base += 64;
        remaining -= 64;
    }

    if remaining > 0 {
        let avail = unsafe { end.offset_from(ptr) } as usize;
        let word = unsafe { load_partial_u64(ptr, avail) };
        let masked = if remaining < 64 {
            word & ((1u64 << remaining) - 1)
        } else {
            word
        };
        dst = unsafe { drain_word_bmi2(masked, base, dst) };
    }

    let written = unsafe { dst.offset_from(out.as_ptr()) } as usize;
    debug_assert_eq!(written, count);
    unsafe { out.set_len(written) };

    out
}

/// Drain set bits using BMI1 BLSR + TZCNT.
///
/// # Safety
/// Caller must ensure `dst` has room for `word.count_ones()` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "bmi1,bmi2")]
unsafe fn drain_word_bmi2(word: u64, base: u32, mut dst: *mut u32) -> *mut u32 {
    use std::arch::x86_64::_blsr_u64;

    if word == 0 {
        return dst;
    }

    if word == u64::MAX {
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
            w = _blsr_u64(w);
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

    #[rstest]
    #[case(1000, 100)] // 10%
    #[case(1000, 20)] // 2%
    #[case(10000, 100)] // 1%
    #[case(10000, 500)] // 5%
    #[case(10000, 2000)] // 20%
    #[case(257, 100)] // odd size near 256-bit boundary
    #[case(512, 50)] // exactly 8 words
    #[case(513, 50)] // 8 words + 1 bit
    fn test_various_densities(#[case] len: usize, #[case] period: usize) {
        let buf = BitBuffer::from_iter((0..len).map(|i| i % period == 0));
        let expected = arrow_set_indices(&buf);
        let actual_iter: Vec<usize> =
            ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len()).collect();
        assert_eq!(
            expected, actual_iter,
            "iterator mismatch for len={len} period={period}"
        );
        let expected_u32: Vec<u32> = expected.iter().map(|&i| i as u32).collect();
        let actual_collect = collect_set_indices(buf.inner().as_slice(), buf.offset(), buf.len());
        assert_eq!(
            expected_u32, actual_collect,
            "collect mismatch for len={len} period={period}"
        );
    }

    #[test]
    fn test_random_pattern() {
        fn splitmix(i: usize) -> u64 {
            let mut z = (i as u64).wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }
        for density in [1u64, 5, 10, 20, 50] {
            let len = 100_000;
            let buf = BitBuffer::from_iter((0..len).map(|i| (splitmix(i) % 100) < density));
            let expected = arrow_set_indices(&buf);
            let actual_iter: Vec<usize> =
                ScalarBitIndexIterator::new(buf.inner().as_slice(), buf.offset(), buf.len())
                    .collect();
            assert_eq!(expected, actual_iter, "random iter mismatch at {density}%");
            let expected_u32: Vec<u32> = expected.iter().map(|&i| i as u32).collect();
            let actual_collect =
                collect_set_indices(buf.inner().as_slice(), buf.offset(), buf.len());
            assert_eq!(
                expected_u32, actual_collect,
                "random collect mismatch at {density}%"
            );
        }
    }
}
