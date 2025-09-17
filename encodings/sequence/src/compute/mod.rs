// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod is_sorted;
mod list_contains;
mod min_max;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_dtype::Nullability;

    use crate::SequenceArray;

    #[rstest]
    // Basic sequence arrays - A[i] = base + i * multiplier
    #[case::sequence_i32(SequenceArray::typed_new(
        0i32,      // base
        1i32,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [0, 1, 2, 3, 4]
    #[case::sequence_i64_step2(SequenceArray::typed_new(
        10i64,     // base
        2i64,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [10, 12, 14, 16, 18]

    // Different types
    #[case::sequence_u32(SequenceArray::typed_new(
        100u32,    // base
        10u32,     // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [100, 110, 120, 130, 140]
    #[case::sequence_i16(SequenceArray::typed_new(
        -10i16,    // base
        3i16,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [-10, -7, -4, -1, 2]

    // Edge cases
    #[case::sequence_single(SequenceArray::typed_new(
        42i32,
        0i32,      // multiplier of 0 means constant array
        Nullability::NonNullable,
        1
    ).unwrap())]
    #[case::sequence_zero_multiplier(SequenceArray::typed_new(
        100i32,
        0i32,      // All values will be 100
        Nullability::NonNullable,
        5
    ).unwrap())]
    #[case::sequence_negative_step(SequenceArray::typed_new(
        100i32,
        -10i32,    // Decreasing sequence
        Nullability::NonNullable,
        5
    ).unwrap())] // Results in [100, 90, 80, 70, 60]

    // Large arrays
    #[case::sequence_large(SequenceArray::typed_new(
        0i64,
        1i64,
        Nullability::NonNullable,
        2000
    ).unwrap())] // Results in [0, 1, 2, ..., 1999]
    #[case::sequence_large_step(SequenceArray::typed_new(
        1000i32,
        100i32,
        Nullability::NonNullable,
        1500
    ).unwrap())] // Results in [1000, 1100, 1200, ..., 150900]

    fn test_sequence_consistency(#[case] array: SequenceArray) {
        test_array_consistency(array.as_ref());
    }
}
