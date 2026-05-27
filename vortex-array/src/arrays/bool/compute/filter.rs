// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::get_bit;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::filter::FilterReduce;

/// Below this density threshold, use the sparse path which iterates only set
/// bits in the mask. Above it, the word-level PEXT approach is faster.
const SPARSE_DENSITY_THRESHOLD: f64 = 0.05;

impl FilterReduce for Bool {
    fn filter(array: ArrayView<'_, Bool>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        let src = array.to_bit_buffer();
        let density = mask_values.density();
        let buffer = if density < SPARSE_DENSITY_THRESHOLD {
            filter_sparse(&src, mask_values, mask.true_count())
        } else {
            filter_bitbuffer_by_mask(&src, mask_values.bit_buffer(), mask.true_count())
        };

        Ok(Some(BoolArray::new(buffer, validity).into_array()))
    }
}

fn filter_sparse(src: &BitBuffer, mask_values: &MaskValues, true_count: usize) -> BitBuffer {
    if let Some(slices) = mask_values.cached_slices() {
        filter_slices(src, true_count, slices.iter().copied())
    } else if let Some(indices) = mask_values.cached_indices() {
        let buffer = src.inner().as_ref();
        let offset = src.offset();
        BitBuffer::collect_bool(indices.len(), |idx| {
            // SAFETY: `collect_bool` calls the closure exactly `indices.len()` times.
            let idx = unsafe { *indices.get_unchecked(idx) };
            get_bit(buffer, offset + idx)
        })
    } else {
        filter_set_bits(src, mask_values.bit_buffer(), true_count)
    }
}

fn filter_slices(
    src: &BitBuffer,
    output_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> BitBuffer {
    let mut builder = BitBufferMut::with_capacity(output_len);
    for (start, end) in slices {
        builder.append_buffer(&src.slice(start..end));
    }
    builder.freeze()
}

fn filter_set_bits(src: &BitBuffer, mask_buf: &BitBuffer, true_count: usize) -> BitBuffer {
    let buffer = src.inner().as_ref();
    let offset = src.offset();
    let mut indices = mask_buf.set_indices();
    BitBuffer::collect_bool(true_count, |_| {
        // SAFETY: the iterator yields exactly true_count indices.
        let idx = unsafe { indices.next().unwrap_unchecked() };
        get_bit(buffer, offset + idx)
    })
}

/// Extract bits from `src` where corresponding bits in `mask_buf` are set.
///
/// Uses a software PEXT (parallel bit extract) to compact selected bits from
/// each 64-bit word, with a u128 accumulator to simplify overflow handling.
/// Fast paths skip PEXT entirely for all-ones and all-zeros mask words.
pub fn filter_bitbuffer_by_mask(
    src: &BitBuffer,
    mask_buf: &BitBuffer,
    true_count: usize,
) -> BitBuffer {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("bmi2") {
            // SAFETY: BMI2 confirmed available; the inner function is compiled with BMI2.
            return unsafe { filter_pext_bmi2(src, mask_buf, true_count) };
        }
    }
    filter_pext_fallback(src, mask_buf, true_count)
}

/// BMI2-native filter: entire function compiled with BMI2+POPCNT enabled.
///
/// The compiler generates PEXT for bit extraction, SHLX/SHRX for flag-free
/// shifts, and POPCNT for population count — no runtime feature checks in
/// the hot loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,popcnt")]
unsafe fn filter_pext_bmi2(src: &BitBuffer, mask_buf: &BitBuffer, true_count: usize) -> BitBuffer {
    use std::arch::x86_64::_pext_u64;

    filter_inner(src, mask_buf, true_count, |src, mask| _pext_u64(src, mask))
}

/// Software fallback filter using byte-LUT PEXT.
fn filter_pext_fallback(src: &BitBuffer, mask_buf: &BitBuffer, true_count: usize) -> BitBuffer {
    filter_inner(src, mask_buf, true_count, pext_fallback)
}

/// Core filter loop parameterized by the PEXT implementation.
///
/// Extracted so the same logic is shared between the software and hardware paths.
/// Uses raw pointer writes instead of Vec::push to eliminate bounds checks
/// in the hot loop — we know the exact output size from true_count.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
fn filter_inner(
    src: &BitBuffer,
    mask_buf: &BitBuffer,
    true_count: usize,
    pext_fn: impl Fn(u64, u64) -> u64,
) -> BitBuffer {
    debug_assert_eq!(src.len(), mask_buf.len());

    let src_chunks = src.chunks();
    let mask_chunks = mask_buf.chunks();

    let out_u64s = true_count.div_ceil(64);
    let mut output: Vec<u64> = Vec::with_capacity(out_u64s + 1);
    let out_ptr = output.as_mut_ptr();
    let mut out_idx: usize = 0;

    // u128 accumulator: overflow naturally held in upper 64 bits, eliminating
    // the tricky `extracted >> (popcount - accum_bits)` re-derivation. Just
    // flush low 64 bits and shift down when full.
    let mut accum: u128 = 0;
    let mut accum_bits: u32 = 0;

    for (src_word, mask_word) in src_chunks.iter().zip(mask_chunks.iter()) {
        if mask_word == u64::MAX {
            // All 64 bits selected — copy source word directly, no PEXT needed.
            accum |= (src_word as u128) << accum_bits;
            accum_bits += 64;
            if accum_bits >= 64 {
                unsafe { out_ptr.add(out_idx).write(accum as u64) };
                out_idx += 1;
                accum >>= 64;
                accum_bits -= 64;
            }
            continue;
        }

        let popcount = mask_word.count_ones();
        if popcount == 0 {
            continue;
        }

        let extracted = pext_fn(src_word, mask_word);

        accum |= (extracted as u128) << accum_bits;
        accum_bits += popcount;

        if accum_bits >= 64 {
            unsafe { out_ptr.add(out_idx).write(accum as u64) };
            out_idx += 1;
            accum >>= 64;
            accum_bits -= 64;
        }
    }

    let remainder = mask_chunks.remainder_bits();
    if remainder != 0 {
        let src_rem = src_chunks.remainder_bits();
        let popcount = remainder.count_ones();
        if popcount > 0 {
            let extracted = pext_fn(src_rem, remainder);
            accum |= (extracted as u128) << accum_bits;
            accum_bits += popcount;
            if accum_bits >= 64 {
                unsafe { out_ptr.add(out_idx).write(accum as u64) };
                out_idx += 1;
                accum >>= 64;
                accum_bits -= 64;
            }
        }
    }

    if accum_bits > 0 {
        unsafe { out_ptr.add(out_idx).write(accum as u64) };
        out_idx += 1;
    }

    // SAFETY: we wrote exactly out_idx words, which is <= out_u64s + 1 = capacity.
    unsafe { output.set_len(out_idx) };

    let byte_len = true_count.div_ceil(8);
    let bytes: Vec<u8> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(output);
        let ptr = v.as_mut_ptr() as *mut u8;
        let cap = v.capacity() * 8;
        Vec::from_raw_parts(ptr, byte_len, cap)
    };

    BitBuffer::new(bytes.into(), true_count)
}

/// Byte-level LUT PEXT fallback.
///
/// Processes each byte of the u64 independently using a precomputed 256-entry
/// lookup table per mask byte. Each byte PEXT is a single table lookup with no
/// data dependencies between bytes, making this faster than the parallel-prefix
/// approach (~12ns vs ~18ns per word).
#[inline(always)]
pub fn pext_fallback(src: u64, mask: u64) -> u64 {
    pext_byte_lut(src, mask)
}

/// Precomputed lookup table for 8-bit PEXT.
///
/// `BYTE_PEXT_LUT[mask_byte]` is a 256-byte table mapping `src_byte` to the
/// extracted bits. Total size: 256 * 256 = 64KB, fits in L1 cache.
#[allow(clippy::cast_possible_truncation)]
static BYTE_PEXT_LUT: &[u8; 256 * 256] = &{
    let mut lut = [0u8; 256 * 256];
    let mut mask: usize = 0;
    while mask < 256 {
        let mut src: usize = 0;
        while src < 256 {
            let mut result = 0u8;
            let mut bit = 0u8;
            // mask and src are always < 256, so truncation to u8 is safe.
            let mut m = mask as u8;
            let s = src as u8;
            let mut pos: u8 = 0;
            while m != 0 {
                if m & 1 != 0 {
                    if s & (1 << pos) != 0 {
                        result |= 1 << bit;
                    }
                    bit += 1;
                }
                m >>= 1;
                pos += 1;
            }
            lut[mask * 256 + src] = result;
            src += 1;
        }
        mask += 1;
    }
    lut
};

/// Byte-level PEXT using precomputed lookup table.
#[inline(always)]
fn pext_byte_lut(src: u64, mask: u64) -> u64 {
    let src_bytes = src.to_le_bytes();
    let mask_bytes = mask.to_le_bytes();

    let mut result: u64 = 0;
    let mut bit_offset: u32 = 0;

    // Unroll the byte loop for performance.
    macro_rules! process_byte {
        ($i:expr) => {
            let m = mask_bytes[$i];
            if m != 0 {
                let extracted = BYTE_PEXT_LUT[(m as usize) * 256 + (src_bytes[$i] as usize)];
                result |= (extracted as u64) << bit_offset;
                bit_offset += m.count_ones();
            }
        };
    }

    process_byte!(0);
    process_byte!(1);
    process_byte!(2);
    process_byte!(3);
    process_byte!(4);
    process_byte!(5);
    process_byte!(6);
    process_byte!(7);

    let _ = bit_offset;
    result
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        let filtered = arr.filter(mask).unwrap();
        assert_arrays_eq!(filtered, BoolArray::from_iter([true, false]));
    }

    #[test]
    fn filter_bool_sparse_index_mask() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_indices(3, [0, 2]);

        let filtered = arr.filter(mask).unwrap();
        assert_arrays_eq!(filtered, BoolArray::from_iter([true, false]));
    }

    #[test]
    fn filter_bool_sparse_slice_mask() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_slices(3, vec![(0, 1), (2, 3)]);

        let filtered = arr.filter(mask).unwrap();
        assert_arrays_eq!(filtered, BoolArray::from_iter([true, false]));
    }

    #[test]
    fn filter_bool_sparse_buffer_mask() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));

        let filtered = arr.filter(mask).unwrap();
        assert_arrays_eq!(filtered, BoolArray::from_iter([true, false]));
    }

    #[test]
    fn filter_bool_by_buffer() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered =
            filter_bitbuffer_by_mask(&arr.to_bit_buffer(), &BitBuffer::from_indices(3, [0, 2]), 2);
        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    #[case(BoolArray::from_iter((0..1024).map(|i| i % 3 != 0)))]
    fn test_filter_bool_conformance(#[case] array: BoolArray) {
        test_filter_conformance(&array.into_array());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_pext_fallback_matches_hardware() {
        use std::arch::x86_64::_pext_u64;

        use super::pext_fallback;

        if !std::arch::is_x86_feature_detected!("bmi2") {
            return;
        }
        let test_cases: Vec<(u64, u64)> = vec![
            (0, 0),
            (u64::MAX, u64::MAX),
            (u64::MAX, 0),
            (0, u64::MAX),
            (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555),
            (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA),
            (0xDEAD_BEEF_CAFE_BABE, 0xFFFF_0000_FFFF_0000),
            (0x1234_5678_9ABC_DEF0, 0xF0F0_F0F0_F0F0_F0F0),
            (u64::MAX, 1),
            (u64::MAX, 1u64 << 63),
            (0x8000_0000_0000_0001, 0x8000_0000_0000_0001),
        ];
        for (src, mask) in test_cases {
            let hw = unsafe { _pext_u64(src, mask) };
            let sw = pext_fallback(src, mask);
            assert_eq!(hw, sw, "mismatch for src={src:#018x} mask={mask:#018x}");
        }
        let mut rng = 0xDEAD_BEEF_u64;
        for _ in 0..1000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let src = rng;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mask = rng;
            let hw = unsafe { _pext_u64(src, mask) };
            let sw = pext_fallback(src, mask);
            assert_eq!(hw, sw, "mismatch for src={src:#018x} mask={mask:#018x}");
        }
    }
}
