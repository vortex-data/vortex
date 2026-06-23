// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
pub(crate) mod compare;
mod filter;
pub(crate) mod is_sorted;
mod list_contains;
pub(crate) mod min_max;
mod slice;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::Nullability;

    use crate::Sequence;
    use crate::SequenceArray;

    #[rstest]
    // Basic sequence arrays - A[i] = base + i * multiplier
    #[case::sequence_i32(Sequence::try_new_typed(
        0i32,      // base
        1i32,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [0, 1, 2, 3, 4]
    #[case::sequence_i64_step2(Sequence::try_new_typed(
        10i64,     // base
        2i64,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [10, 12, 14, 16, 18]

    // Different types
    #[case::sequence_u32(Sequence::try_new_typed(
        100u32,    // base
        10u32,     // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [100, 110, 120, 130, 140]
    #[case::sequence_i16(Sequence::try_new_typed(
        -10i16,    // base
        3i16,      // multiplier
        Nullability::NonNullable,
        5          // length
    ).unwrap())] // Results in [-10, -7, -4, -1, 2]

    // Edge cases
    #[case::sequence_single(Sequence::try_new_typed(
        42i32,
        0i32,      // multiplier of 0 means constant array
        Nullability::NonNullable,
        1
    ).unwrap())]
    #[case::sequence_zero_multiplier(Sequence::try_new_typed(
        100i32,
        0i32,      // All values will be 100
        Nullability::NonNullable,
        5
    ).unwrap())]
    #[case::sequence_negative_step(Sequence::try_new_typed(
        100i32,
        -10i32,    // Decreasing sequence
        Nullability::NonNullable,
        5
    ).unwrap())] // Results in [100, 90, 80, 70, 60]

    // Large arrays
    #[case::sequence_large(Sequence::try_new_typed(
        0i64,
        1i64,
        Nullability::NonNullable,
        2000
    ).unwrap())] // Results in [0, 1, 2, ..., 1999]
    #[case::sequence_large_step(Sequence::try_new_typed(
        1000i32,
        100i32,
        Nullability::NonNullable,
        1500
    ).unwrap())] // Results in [1000, 1100, 1200, ..., 150900]

    fn test_sequence_consistency(#[case] array: SequenceArray) {
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
