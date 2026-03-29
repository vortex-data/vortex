// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::RunEndData;
mod cast;
mod compare;
mod fill_null;
pub(crate) mod filter;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
pub(crate) mod min_max;
pub(crate) mod take;
pub(crate) mod take_from;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;

    use crate::RunEndArray;

    #[rstest]
    // Simple run-end arrays
    #[case::runend_i32(RunEndData::encode(
        buffer![1i32, 1, 1, 2, 2, 3, 3, 3, 3].into_array()
    ).unwrap())]
    #[case::runend_single_run(RunEndData::encode(
        buffer![5i32, 5, 5, 5, 5].into_array()
    ).unwrap())]
    #[case::runend_alternating(RunEndData::encode(
        buffer![1i32, 2, 1, 2, 1, 2].into_array()
    ).unwrap())]
    // Different types
    #[case::runend_u64(RunEndData::encode(
        buffer![100u64, 100, 200, 200, 200].into_array()
    ).unwrap())]
    // Edge cases
    #[case::runend_single(RunEndData::encode(
        buffer![42i32].into_array()
    ).unwrap())]
    #[case::runend_large(RunEndData::encode(
        PrimitiveArray::from_iter((0..1000).map(|i| i / 10)).into_array()
    ).unwrap())]

    fn test_runend_consistency(#[case] array: RunEndArray) {
        test_array_consistency(&array.into_array());
    }
}
