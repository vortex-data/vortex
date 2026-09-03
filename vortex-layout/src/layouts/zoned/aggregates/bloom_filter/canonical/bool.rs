// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use super::BloomPartial;

/// Similar to `accumulate_bool` implementation in [vortex_array::aggregate_fn::fns::min_max]
pub(super) fn accumulate_bool(
    array: &BoolArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;

    let (true_count, valid_count) = match mask.bit_buffer() {
        AllOr::None => return Ok(()),
        AllOr::All => (array.bit_buffer_view().true_count(), array.as_ref().len()),
        AllOr::Some(validity) => (
            array.to_bit_buffer().bitand(validity).true_count(),
            validity.true_count(),
        ),
    };

    if true_count > 0 {
        // True present
        partial.insert([0x1]);
    }
    if true_count < valid_count {
        // False present
        partial.insert([0x0]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::build_filter;
    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::setup;

    #[rstest]
    #[case::inserts_each_valid_boolean_value(
        &[Some(true), Some(false)],
        Nullability::NonNullable,
        true,
        true
    )]
    #[case::inserts_only_false(
        &[Some(false), Some(false)],
        Nullability::NonNullable,
        false,
        true
    )]
    #[case::inserts_only_true(
        &[Some(true), Some(true)],
        Nullability::NonNullable,
        true,
        false
    )]
    #[case::ignores_null_boolean_values(
        &[Some(true), None, Some(true)],
        Nullability::Nullable,
        true,
        false
    )]
    #[case::all_null_booleans_leave_the_filter_empty(
        &[None, None],
        Nullability::Nullable,
        false,
        false
    )]
    fn membership(
        #[case] values: &[Option<bool>],
        #[case] nullability: Nullability,
        #[case] expect_true: bool,
        #[case] expect_false: bool,
    ) -> VortexResult<()> {
        let ctx = setup()?;
        let array = match nullability {
            Nullability::NonNullable => BoolArray::from_iter(
                values
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| vortex_err!("non-null test case contains a null"))?,
            ),
            Nullability::Nullable => BoolArray::from_iter(values.iter().copied()),
        };
        let bloom_filter = build_filter(array.into_array(), DType::Bool(nullability), ctx)?;

        assert_eq!(
            bloom_filter.contains_scalar(&Scalar::bool(true, nullability))?,
            expect_true
        );
        assert_eq!(
            bloom_filter.contains_scalar(&Scalar::bool(false, nullability))?,
            expect_false
        );

        Ok(())
    }
}
