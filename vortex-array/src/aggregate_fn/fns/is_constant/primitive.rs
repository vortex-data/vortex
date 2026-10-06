// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "avx2")] {
        pub const IS_CONST_LANE_WIDTH: usize = 32;
    } else {
        pub const IS_CONST_LANE_WIDTH: usize = 16;
    }
}

/// Assumes any floating point has been cast into its bit representation for which != and !is_eq are the same
/// Assumes there's at least 1 value in the slice, which is an invariant of the entry level function.
pub fn compute_is_constant<T: NativePType, const WIDTH: usize>(values: &[T]) -> bool {
    let first_value = values[0];
    let first_vec = &[first_value; WIDTH];

    let (chunks, remainder) = values[1..].as_chunks::<WIDTH>();
    for chunk in chunks {
        if first_vec != chunk {
            return false;
        }
    }

    for value in remainder {
        if !value.is_eq(first_value) {
            return false;
        }
    }

    true
}

trait EqFloat {
    type IntType;
}

impl EqFloat for f16 {
    type IntType = u16;
}
impl EqFloat for f32 {
    type IntType = u32;
}
impl EqFloat for f64 {
    type IntType = u64;
}

pub(super) fn check_primitive_constant(array: &PrimitiveArray) -> bool {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if array.len() * array.ptype().byte_width() >= simd::MIN_BYTES {
        return match_each_native_ptype!(array.ptype(), |P| {
            let values = array.as_slice::<P>();
            // SAFETY: native ptypes have no padding, so every byte of the slice is initialized.
            let bytes = unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
            };
            simd::is_constant_bytes::<{ size_of::<P>() }>(bytes)
        });
    }

    match_each_native_ptype!(array.ptype(), integral: |P| {
        compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<P>()}>(array.as_slice::<P>())
    }, floating: |P| {
        compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<P>()}>(unsafe { std::mem::transmute::<&[P], &[<P as EqFloat>::IntType]>(array.as_slice::<P>()) })
    })
}

/// Runtime-dispatched bitwise comparison over the raw bytes of a primitive slice, which covers
/// every element width (and float bit patterns) with one 64-byte kernel.
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod simd {
    use std::sync::LazyLock;

    use fearless_simd::Level;
    use fearless_simd::Simd;
    use fearless_simd::dispatch;
    use fearless_simd::prelude::*;
    use fearless_simd::u8x64;
    use fearless_simd::u64x8;

    /// Below this the dispatch costs more than the compiler-vectorized comparison saves.
    pub(super) const MIN_BYTES: usize = 256;

    /// Detected once: `Level::new` probes every feature of its widest level on each call.
    static SIMD_LEVEL: LazyLock<Level> = LazyLock::new(Level::new);

    /// Whether every `ELEM`-byte element of `bytes` has the bit pattern of the first.
    pub(super) fn is_constant_bytes<const ELEM: usize>(bytes: &[u8]) -> bool {
        debug_assert!(matches!(ELEM, 1 | 2 | 4 | 8));
        debug_assert_eq!(bytes.len() % ELEM, 0);
        if bytes.len() <= ELEM {
            return true;
        }
        let word = u64::from_ne_bytes(std::array::from_fn(|i| bytes[i % ELEM]));
        dispatch!(*SIMD_LEVEL, simd => is_constant_kernel(simd, bytes, word))
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_constant_kernel<S: Simd>(simd: S, bytes: &[u8], word: u64) -> bool {
        let splat: u8x64<S> = u64x8::splat(simd, word).bitcast();
        let zero = u8x64::splat(simd, 0);

        // Fold four vectors per early-exit check.
        let (blocks, rest) = bytes.as_chunks::<256>();
        for block in blocks {
            let (vectors, _) = block.as_chunks::<64>();
            let diff = (u8x64::from_slice(simd, &vectors[0]) ^ splat)
                | (u8x64::from_slice(simd, &vectors[1]) ^ splat)
                | (u8x64::from_slice(simd, &vectors[2]) ^ splat)
                | (u8x64::from_slice(simd, &vectors[3]) ^ splat);
            if !diff.simd_eq(zero).all_true() {
                return false;
            }
        }

        let (vectors, tail) = rest.as_chunks::<64>();
        if !vectors
            .iter()
            .all(|vector| u8x64::from_slice(simd, vector).simd_eq(splat).all_true())
        {
            return false;
        }

        // 64 is a multiple of every element width, so the tail starts on an element boundary.
        let pattern = word.to_ne_bytes();
        tail.chunks(8).all(|chunk| chunk == &pattern[..chunk.len()])
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::check_primitive_constant;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::NativePType;

    /// Covers both sides of the SIMD size threshold, every chunking remainder, and a
    /// difference in the first, middle and last element.
    fn check<T: NativePType>(base: T, other: T) {
        for len in [1usize, 2, 63, 64, 65, 255, 256, 257, 1000, 4099] {
            let constant = PrimitiveArray::from_iter(vec![base; len]);
            assert!(check_primitive_constant(&constant), "len {len}");
            for pos in [0, len / 2, len - 1] {
                if len < 2 {
                    continue;
                }
                let mut values = vec![base; len];
                values[pos] = other;
                let array = PrimitiveArray::from_iter(values);
                assert!(!check_primitive_constant(&array), "len {len} pos {pos}");
            }
        }
    }

    #[rstest]
    fn is_constant_all_widths() {
        check::<u8>(7, 9);
        check::<i16>(-7, 9);
        check::<u32>(7, 9);
        check::<i64>(-7, 9);
        check::<f32>(1.5, 2.5);
        check::<f64>(1.5, -1.5);
    }

    #[test]
    fn is_constant_distinguishes_float_bit_patterns() {
        let mut values = vec![0.0f64; 300];
        values[150] = -0.0;
        assert!(!check_primitive_constant(&PrimitiveArray::from_iter(values)));
    }
}
