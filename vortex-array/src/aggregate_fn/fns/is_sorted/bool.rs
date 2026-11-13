// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedOptions;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;

pub(super) fn check_bool_sorted(
    array: &BoolArray,
    options: &IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllFalse(_) => Ok(!options.strict),
        Mask::AllTrue(_) => {
            let values = array.bit_buffer_view();
            Ok(options.is_sorted(values.iter()))
        }
        Mask::Values(mask_values) => {
            // TODO (joacoc): `primitive`, `decimal` and `bool` repeat the same match on the validity
            // mask, and now it could consolidate into a single `is_masked_sorted`.
            if options.strict {
                let validity_buffer = mask_values.bit_buffer();
                let values = array.bit_buffer_view();
                Ok(options.is_nullable_sorted(
                    validity_buffer
                        .iter()
                        .zip(values.iter())
                        .map(|(is_valid, value)| is_valid.then_some(value)),
                ))
            } else {
                // TODO (joacoc): Here nulls are skipped, but everywhere else they come first.
                // For example, `[false, null, true]` is sorted here because the null is skipped,
                // but the same pattern with integers, `[0, null, 1]`, is not sorted.
                // The checks between batches also put nulls first, so `[true, null]` is sorted as
                // one batch, but not when split into `[true]` and `[null]`.
                let set_indices = mask_values.bit_buffer().set_indices();
                let values = array.bit_buffer_view();
                let values_iter = set_indices.map(|idx|
                    // Safety:
                    // All idxs are in-bounds for the array.
                    unsafe {
                        values.value_unchecked(idx)
                    });
                Ok(options.is_sorted(values_iter))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::check_bool_sorted;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;

    /// Booleans skip nulls when checking non-strict order, so these arrays count as sorted.
    /// Integers with nulls in the same spots don't, because their nulls have to come first.
    /// This is the mismatch described in the TODO above. Once it's fixed, we can remove test.
    #[rstest]
    #[case::interior_null([Some(false), None, Some(true)], [Some(0i32), None, Some(1)])]
    #[case::trailing_null([Some(false), Some(true), None], [Some(0i32), Some(1), None])]
    fn non_strict_skips_nulls(
        #[case] booleans: [Option<bool>; 3],
        #[case] integers: [Option<i32>; 3],
    ) -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();

        // Only `[false, true]` reaches the check, which is sorted.
        let booleans = BoolArray::from_iter(booleans);
        assert!(check_bool_sorted(
            &booleans,
            &IsSortedOptions::default(),
            &mut ctx
        )?);

        let integers = PrimitiveArray::from_option_iter(integers).into_array();
        assert!(!is_sorted(&integers, &mut ctx)?);
        Ok(())
    }
}
