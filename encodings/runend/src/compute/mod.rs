// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod binary_numeric;
mod compare;
mod fill_null;
pub(crate) mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod min_max;
pub(crate) mod take;
mod take_from;

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn test_runend_binary_numeric() {
        let array = ree_array().into_array();
        test_binary_numeric_array(array)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;

    use crate::RunEndArray;

    #[rstest]
    // Simple run-end arrays
    #[case::runend_i32(RunEndArray::encode(
        PrimitiveArray::from_iter([1i32, 1, 1, 2, 2, 3, 3, 3, 3]).into_array()
    ).unwrap())]
    #[case::runend_single_run(RunEndArray::encode(
        PrimitiveArray::from_iter([5i32, 5, 5, 5, 5]).into_array()
    ).unwrap())]
    #[case::runend_alternating(RunEndArray::encode(
        PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2]).into_array()
    ).unwrap())]
    // Different types
    #[case::runend_u64(RunEndArray::encode(
        PrimitiveArray::from_iter([100u64, 100, 200, 200, 200]).into_array()
    ).unwrap())]
    // Edge cases
    #[case::runend_single(RunEndArray::encode(
        PrimitiveArray::from_iter([42i32]).into_array()
    ).unwrap())]
    #[case::runend_large(RunEndArray::encode(
        PrimitiveArray::from_iter((0..1000).map(|i| i / 10)).into_array()
    ).unwrap())]

    fn test_runend_consistency(#[case] array: RunEndArray) {
        test_array_consistency(array.as_ref());
    }

    #[rstest]
    #[case::runend_i32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([10i32, 10, 10, 20, 20, 30, 30, 30, 30]).into_array()
    ).unwrap())]
    #[case::runend_u32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([100u32, 100, 200, 200, 200]).into_array()
    ).unwrap())]
    #[case::runend_i64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([1000i64, 1000, 2000, 2000, 3000, 3000]).into_array()
    ).unwrap())]
    #[case::runend_u64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([5000u64, 5000, 5000, 6000, 6000]).into_array()
    ).unwrap())]
    #[case::runend_f32_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([1.5f32, 1.5, 2.5, 2.5, 3.5]).into_array()
    ).unwrap())]
    #[case::runend_f64_basic(RunEndArray::encode(
        PrimitiveArray::from_iter([10.1f64, 10.1, 20.2, 20.2, 20.2]).into_array()
    ).unwrap())]
    #[case::runend_i32_large(RunEndArray::encode(
        PrimitiveArray::from_iter((0..100).map(|i| i / 5)).into_array()
    ).unwrap())]
    fn test_runend_binary_numeric(#[case] array: RunEndArray) {
        test_binary_numeric_array(array.into_array());
    }
}
