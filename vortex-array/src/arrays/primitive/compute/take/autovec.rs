// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An auto-vectorized implementation of the take kernel for primitive arrays.
//!
//! Uses `multiversion` for runtime CPU feature detection and dispatch so that the compiler
//! can emit gather instructions (e.g. `vpgatherdd` on AVX2, `vpgatherdd zmm` on AVX-512)
//! without hand-written intrinsics.

use multiversion::multiversion;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

pub(super) struct TakeKernelAutoVec;

impl TakeImpl for TakeKernelAutoVec {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        Ok(match_each_native_ptype!(array.ptype(), |V| {
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                let buffer = take_autovec(array.as_slice::<V>(), indices.as_slice::<I>());
                PrimitiveArray::new(buffer, validity).into_array()
            })
        }))
    }
}

/// Auto-vectorized take: gathers `values[indices[i]]` for each index.
///
/// The `multiversion` attribute compiles this function for multiple targets and dispatches
/// to the best available at runtime. With AVX2/AVX-512 enabled, the compiler can emit
/// hardware gather instructions for the inner loop.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw",
    "x86_64+avx2",
    "x86_64+sse4.2",
    "aarch64+neon",
))]
fn take_autovec<V: NativePType, I: UnsignedPType>(
    values: &[V],
    indices: &[I],
) -> Buffer<V> {
    let len = indices.len();
    let values_len = values.len();

    let mut buffer = BufferMut::<V>::with_capacity(len);
    let buf = buffer.spare_capacity_mut();

    let src = values.as_ptr();
    let dst = buf.as_mut_ptr().cast::<V>();

    // Fused bounds-check + gather loop.
    //
    // The per-element `idx < values_len` comparison generates a mask that the compiler
    // folds directly into a masked gather instruction on AVX-512 (vpgatherdd {k}, ...)
    // and into conditional moves on AVX2. Out-of-bounds lanes write zero (V::default()).
    //
    // We track the running maximum index so we can report a precise panic after the loop
    // without adding a branch to the hot path. The max reduction itself vectorizes into
    // vpmaxud / umaxv instructions.
    let mut max_seen: usize = 0;

    for i in 0..len {
        let idx: usize = unsafe { (*indices.as_ptr().add(i)).as_() };
        max_seen = max_seen.max(idx);

        // SAFETY: `i < len` so dst write is in-bounds.
        // The branch on `idx < values_len` becomes a mask for the gather; the compiler
        // emits vpgatherdd {k1} / vpgatherdq {k1} with the comparison result as the mask.
        unsafe {
            let val = if idx < values_len {
                *src.add(idx)
            } else {
                V::default()
            };
            dst.add(i).write(val);
        }
    }

    assert!(
        max_seen < values_len || len == 0,
        "take index {max_seen} out of bounds for array of length {values_len}"
    );

    // SAFETY: We wrote exactly `len` elements above.
    unsafe { buffer.set_len(len) };
    buffer.freeze()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::*;
    use crate::arrays::BoolArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    #[test]
    fn test_take_autovec_simple() {
        let values = vec![10i32, 20, 30, 40, 50];
        let indices = vec![0u32, 0, 4, 2];
        let result = take_autovec(&values, &indices);
        assert_eq!(result.as_slice(), &[10i32, 10, 50, 30]);
    }

    #[test]
    fn test_take_autovec_u8_indices() {
        let values: Vec<i32> = (1..=127).collect();
        let indices: Vec<u8> = (0..127).collect();
        let result = take_autovec(&values, &indices);
        assert_eq!(result.as_slice(), &values);
    }

    #[test]
    fn test_take_autovec_u64_values() {
        let values: Vec<u64> = (100..200).collect();
        let indices = vec![0u32, 50, 99, 25];
        let result = take_autovec(&values, &indices);
        assert_eq!(result.as_slice(), &[100u64, 150, 199, 125]);
    }

    #[test]
    fn test_take_autovec_f32() {
        let values = vec![1.0f32, 2.5, 3.7, 4.2, 5.9];
        let indices = vec![4u16, 2, 0, 1, 3];
        let result = take_autovec(&values, &indices);
        assert_eq!(result.as_slice(), &[5.9f32, 3.7, 1.0, 2.5, 4.2]);
    }

    #[test]
    fn test_take_autovec_f64() {
        let values: Vec<f64> = (0..256).map(|i| i as f64 * 1.5).collect();
        let indices: Vec<u32> = (0..256).rev().collect();
        let result = take_autovec(&values, &indices);
        for (i, &v) in result.as_slice().iter().enumerate() {
            assert_eq!(v, (255 - i) as f64 * 1.5);
        }
    }

    #[test]
    fn test_take_autovec_remainder() {
        // 13 elements: 1 full chunk of 8 + 5 remainder
        let values: Vec<i32> = (0..100).collect();
        let indices: Vec<u32> = (0..13).collect();
        let result = take_autovec(&values, &indices);
        let expected: Vec<i32> = (0..13).collect();
        assert_eq!(result.as_slice(), &expected);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_take_autovec_out_of_bounds() {
        let values = vec![1i32, 2, 3];
        let indices = vec![0u32, 5]; // index 5 is out of bounds
        take_autovec(&values, &indices);
    }

    #[test]
    fn test_take_autovec_empty_indices() {
        let values = vec![1i32, 2, 3];
        let indices: Vec<u32> = vec![];
        let result = take_autovec(&values, &indices);
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_take_autovec_large() {
        let values: Vec<i32> = (0..10_000).collect();
        let indices: Vec<u32> = (0..10_000).rev().collect();
        let result = take_autovec(&values, &indices);
        for (i, &v) in result.as_slice().iter().enumerate() {
            assert_eq!(v, (9_999 - i) as i32);
        }
    }

    #[rstest]
    #[case(PrimitiveArray::new(buffer![42i32], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid))]
    #[case(PrimitiveArray::new(
        buffer![0, 1, 2, 3, 4, 5],
        Validity::Array(BoolArray::from_iter([true, false, true, false, true, true]).into_array()),
    ))]
    #[case(PrimitiveArray::from_option_iter([Some(1), None, Some(3), Some(4), None]))]
    fn test_take_autovec_conformance(#[case] array: PrimitiveArray) {
        test_take_conformance(&array.into_array());
    }
}
