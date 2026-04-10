// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod cast;
mod compare;
mod filter;
mod mask;
pub(crate) mod nan_count;
mod slice;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;

    use crate::ALPArray;
    use crate::alp_encode;

    #[rstest]
    // Basic float arrays
    #[case::f32_array(alp_encode(PrimitiveArray::from_iter([1.23f32, 4.56, 7.89, 10.11, 12.13]).as_view(), None).unwrap())]
    #[case::f64_array(alp_encode(PrimitiveArray::from_iter([100.1f64, 200.2, 300.3, 400.4, 500.5]).as_view(), None).unwrap())]
    // Nullable arrays
    #[case::nullable_f32(alp_encode(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).as_view(), None).unwrap())]
    #[case::nullable_f64(alp_encode(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(2.2), Some(3.3), None]).as_view(), None).unwrap())]
    // Edge cases
    #[case::single_element(alp_encode(PrimitiveArray::from_iter([42.42f64]).as_view(), None).unwrap())]
    // Large arrays
    #[case::large_f32(alp_encode(PrimitiveArray::from_iter((0..1000).map(|i| i as f32 * 0.1)).as_view(), None).unwrap())]
    // Arrays with patterns
    #[case::repeating_pattern(alp_encode(PrimitiveArray::from_iter([1.1f32, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3]).as_view(), None).unwrap())]
    #[case::close_values(alp_encode(PrimitiveArray::from_iter([100.001f64, 100.002, 100.003, 100.004, 100.005]).as_view(), None).unwrap())]
    fn test_alp_consistency(#[case] array: ALPArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::f32_basic(alp_encode(PrimitiveArray::from_iter([1.23f32, 4.56, 7.89, 10.11, 12.13]).as_view(), None).unwrap())]
    #[case::f64_basic(alp_encode(PrimitiveArray::from_iter([100.1f64, 200.2, 300.3, 400.4, 500.5]).as_view(), None).unwrap())]
    #[case::f32_large(alp_encode(PrimitiveArray::from_iter((0..100).map(|i| i as f32 * 1.5)).as_view(), None).unwrap())]
    #[case::f64_large(alp_encode(PrimitiveArray::from_iter((0..100).map(|i| i as f64 * 2.5)).as_view(), None).unwrap())]
    fn test_alp_binary_numeric(#[case] array: ALPArray) {
        test_binary_numeric_array(array.into_array());
    }
}
