// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(target_arch = "x86_64")]
use vortex_error::VortexExpect;

#[inline]
pub fn count_ones(bytes: &[u8], offset: usize, len: usize) -> usize {
    if bytes.is_empty() {
        return 0;
    }

    let (head, middle, tail) = align_offset_len(bytes, offset, len);

    let mut count = head.map_or(0, |v| v.count_ones() as usize);

    if !middle.is_empty() {
        count += count_ones_aligned(middle);
    }

    count + tail.map_or(0, |v| v.count_ones() as usize)
}

#[inline]
pub(super) fn align_offset_len(
    bytes: &[u8],
    offset: usize,
    len: usize,
) -> (Option<u8>, &[u8], Option<u8>) {
    let start_byte = offset / 8;
    let start_bit = offset % 8;
    let end_bit = offset + len;
    let end_byte = end_bit / 8;
    let head = (start_bit != 0).then(|| {
        let start_len = (8 - start_bit).min(len);
        mask_byte(bytes[start_byte], start_bit, start_len)
    });

    let middle_start = start_byte + usize::from(start_bit != 0);
    let middle_end = end_byte;
    let middle = if middle_start < middle_end {
        &bytes[middle_start..middle_end]
    } else {
        &[]
    };

    let consumed = if start_bit != 0 {
        (8 - start_bit).min(len)
    } else {
        0
    } + middle.len() * 8;
    let tail_len = len - consumed;
    let tail = (tail_len != 0).then(|| mask_byte(bytes[middle_end], 0, tail_len));

    (head, middle, tail)
}

#[inline]
fn mask_byte(byte: u8, bit_offset: usize, bit_len: usize) -> u8 {
    debug_assert!(bit_offset < 8);
    debug_assert!(bit_len <= 8 - bit_offset);

    let shifted = byte >> bit_offset;
    let mask = if bit_len == 8 {
        u8::MAX
    } else {
        (1u8 << bit_len) - 1
    };

    shifted & mask
}

#[inline]
fn count_ones_aligned(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if bytes.len() >= 64
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: Runtime detection guarantees the required target features.
            return unsafe { count_ones_aligned_avx512(bytes) };
        }

        if bytes.len() >= 32 && is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime detection guarantees the required target features.
            return unsafe { count_ones_aligned_avx2(bytes) };
        }
    }

    count_ones_aligned_scalar(bytes)
}

#[inline]
fn count_ones_aligned_scalar(bytes: &[u8]) -> usize {
    let (words, tail) = bytes.as_chunks::<8>();
    let count = words
        .iter()
        .map(|word| u64::from_le_bytes(*word).count_ones() as usize)
        .sum::<usize>();

    count
        + tail
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_ones_aligned_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_mm256_add_epi8;
    use std::arch::x86_64::_mm256_add_epi64;
    use std::arch::x86_64::_mm256_and_si256;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_sad_epu8;
    use std::arch::x86_64::_mm256_set1_epi8;
    use std::arch::x86_64::_mm256_setr_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;
    use std::arch::x86_64::_mm256_shuffle_epi8;
    use std::arch::x86_64::_mm256_srli_epi16;
    use std::arch::x86_64::_mm256_storeu_si256;

    #[inline]
    unsafe fn byte_popcount(chunk: __m256i, mask: __m256i, lookup: __m256i) -> __m256i {
        let lo = unsafe { _mm256_and_si256(chunk, mask) };
        let hi = unsafe { _mm256_and_si256(_mm256_srli_epi16(chunk, 4), mask) };
        unsafe {
            _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            )
        }
    }

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let mask = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let mut accum = _mm256_setzero_si256();
    let mut index = 0;

    while index + 128 <= bytes.len() {
        for lane in 0..4 {
            let ptr = unsafe { bytes.as_ptr().add(index + lane * 32) }.cast::<__m256i>();
            let chunk = unsafe { _mm256_loadu_si256(ptr) };
            let counts = unsafe { byte_popcount(chunk, mask, lookup) };
            accum = _mm256_add_epi64(accum, _mm256_sad_epu8(counts, zero));
        }
        index += 128;
    }

    while index + 32 <= bytes.len() {
        let ptr = unsafe { bytes.as_ptr().add(index) }.cast::<__m256i>();
        let chunk = unsafe { _mm256_loadu_si256(ptr) };
        let counts = unsafe { byte_popcount(chunk, mask, lookup) };
        accum = _mm256_add_epi64(accum, _mm256_sad_epu8(counts, zero));
        index += 32;
    }

    let mut lanes = [0u64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), accum) };

    usize::try_from(lanes.iter().sum::<u64>()).vortex_expect("true_count doesn't fit in usize")
        + count_ones_aligned_scalar(&bytes[index..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn count_ones_aligned_avx512(bytes: &[u8]) -> usize {
    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm512_add_epi64;
    use std::arch::x86_64::_mm512_loadu_si512;
    use std::arch::x86_64::_mm512_popcnt_epi64;
    use std::arch::x86_64::_mm512_setzero_si512;
    use std::arch::x86_64::_mm512_storeu_si512;

    let mut accum = _mm512_setzero_si512();
    let mut index = 0;

    while index + 64 <= bytes.len() {
        let ptr = unsafe { bytes.as_ptr().add(index) }.cast::<__m512i>();
        let chunk = unsafe { _mm512_loadu_si512(ptr) };
        accum = _mm512_add_epi64(accum, _mm512_popcnt_epi64(chunk));
        index += 64;
    }

    let mut lanes = [0u64; 8];
    unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), accum) };

    usize::try_from(lanes.iter().sum::<u64>()).vortex_expect("true_count doesn't fit in usize")
        + count_ones_aligned_scalar(&bytes[index..])
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::BitBuffer;

    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn test_count_ones_matches_iteration_for_slices(
        #[values(
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30
        )]
        offset: usize,
        #[values(
            0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 513
        )]
        slice_len: usize,
    ) {
        let len = 513;
        let buf = BitBuffer::collect_bool(len + 31, |i| (i % 3 == 0) ^ (i % 11 == 0));

        if offset + slice_len > buf.len() {
            return;
        }

        let sliced = buf.slice(offset..offset + slice_len);
        let expected = sliced.iter().filter(|bit| *bit).count();

        assert_eq!(
            sliced.true_count(),
            expected,
            "offset={offset} len={slice_len}"
        );
    }
}
