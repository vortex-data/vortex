// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod cast;
mod fill_null;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod nan_count;
pub(crate) mod rules;
mod slice;
mod take;

pub use is_constant::*;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::consistency::test_array_consistency;

    #[rstest]
    // Basic primitive arrays
    #[case::i32(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case::nullable_i32(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]))]
    #[case::i64_range(PrimitiveArray::from_iter(0..100i64))]
    #[case::f32(PrimitiveArray::from_iter([1.1f32, 2.2, 3.3, 4.4, 5.5]))]
    #[case::nullable_f64(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(3.3), Some(4.4), None]))]
    // Edge cases
    #[case::single_element(PrimitiveArray::from_iter([42i32]))]
    #[case::two_elements(PrimitiveArray::from_iter([1u64, 2]))]
    // Large arrays
    #[case::large_u32(PrimitiveArray::from_iter(0..2000u32))]
    #[case::large_nullable(PrimitiveArray::from_option_iter((0..2000).map(|i| if i % 7 == 0 { None } else { Some(i as i64) })))]
    // Different numeric types
    #[case::u8(PrimitiveArray::from_iter([0u8, 1, 2, 255]))]
    #[case::i16_negative(PrimitiveArray::from_iter([-100i16, -50, 0, 50, 100]))]
    #[case::f64_special(PrimitiveArray::from_iter([0.0f64, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY]))]
    fn test_primitive_consistency(#[case] array: PrimitiveArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::i32(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case::nullable_i32(PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4), Some(5)]))]
    #[case::i64(PrimitiveArray::from_iter([10i64, 20, 30, 40, 50]))]
    #[case::f32(PrimitiveArray::from_iter([1.5f32, 2.5, 3.5, 4.5, 5.5]))]
    #[case::f64(PrimitiveArray::from_iter([10.1f64, 20.2, 30.3, 40.4, 50.5]))]
    fn test_primitive_binary_numeric(#[case] array: PrimitiveArray) {
        use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
        test_binary_numeric_array(array.into_array());
    }
}
