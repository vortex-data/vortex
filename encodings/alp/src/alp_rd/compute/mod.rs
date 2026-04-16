// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
mod mask;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;

    use crate::ALPRDArray;
    use crate::RDEncoder;

    #[rstest]
    // Basic float arrays with RD (reference delta) encoding
    #[case::f32_array({
        let values = vec![1.0f32, 1.1, 1.2, 1.3, 1.4];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    #[case::f64_array({
        let values = vec![100.0f64, 100.01, 100.02, 100.03, 100.04];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    // Nullable arrays
    #[case::nullable_f32({
        let values = vec![1.0f32, 1.2, 1.3];
        let arr = PrimitiveArray::from_option_iter([Some(1.0f32), None, Some(1.2), Some(1.3), None]);
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    #[case::nullable_f64({
        let values = vec![10.0f64, 10.2, 10.3];
        let arr = PrimitiveArray::from_option_iter([Some(10.0f64), None, Some(10.2), Some(10.3), None]);
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    // Edge cases
    #[case::single_element({
        let values = vec![42.42f64];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    // Arrays with small deltas (good for RD encoding)
    #[case::small_deltas({
        let values = vec![1000.0f32, 1000.001, 1000.002, 1000.003, 1000.004];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    // Large arrays
    #[case::large_f32({
        let values: Vec<f32> = (0..1000).map(|i| 100.0 + i as f32 * 0.01).collect();
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    fn test_alp_rd_consistency(#[case] array: ALPRDArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::f32_basic({
        let values = vec![1.0f32, 1.1, 1.2, 1.3, 1.4];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    #[case::f64_basic({
        let values = vec![100.0f64, 100.01, 100.02, 100.03, 100.04];
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    #[case::f32_large({
        let values: Vec<f32> = (0..100).map(|i| 50.0 + i as f32 * 0.1).collect();
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    #[case::f64_large({
        let values: Vec<f64> = (0..100).map(|i| 1000.0 + i as f64 * 0.01).collect();
        let arr = PrimitiveArray::from_iter(values.clone());
        let encoder = RDEncoder::new(&values);
        encoder.encode(arr.as_view())
    })]
    fn test_alp_rd_binary_numeric(#[case] array: ALPRDArray) {
        test_binary_numeric_array(array.into_array());
    }
}
