// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod fill_null;
pub(crate) mod filter;
mod mask;
pub mod rules;
mod slice;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::compute::conformance::consistency::test_array_consistency;

    #[rstest]
    // Basic bool arrays
    #[case::bool_array(BoolArray::from_iter([true, false, true, true, false]))]
    #[case::nullable_bool(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    // Edge cases
    #[case::single_true(BoolArray::from_iter([true]))]
    #[case::single_false(BoolArray::from_iter([false]))]
    #[case::two_elements(BoolArray::from_iter([true, false]))]
    #[case::all_true(BoolArray::from_iter([true, true, true, true, true]))]
    #[case::all_false(BoolArray::from_iter([false, false, false, false, false]))]
    // Large arrays
    #[case::large_alternating(BoolArray::from_iter((0..2000).map(|i| i % 2 == 0)))]
    #[case::large_sparse_true(BoolArray::from_iter((0..2000).map(|i| i % 100 == 0)))]
    #[case::large_nullable(BoolArray::from_iter((0..2000).map(|i| (i % 10 == 0).then_some(i % 3 == 0))))]
    // Patterns
    #[case::runs_pattern(BoolArray::from_iter(
        [true, true, true, false, false, false, true, true, true, false, false, false]
    ))]
    #[case::mostly_null(BoolArray::from_iter([
        None, None, Some(true), None, None, None, Some(false), None, None
    ]))]
    fn test_bool_consistency(#[case] array: BoolArray) {
        test_array_consistency(&array.into_array());
    }
}
