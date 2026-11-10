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
            // Nulls sort first, so map each element to `Option` where `None < Some(_)`.
            let values = array.bit_buffer_view();
            let iter = mask_values
                .bit_buffer()
                .iter()
                .zip(values.iter())
                .map(|(is_valid, value)| is_valid.then_some(value));
            Ok(options.is_nullable_sorted(iter))
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

    /// Non-strict Boolean checks only read the valid rows, so nulls are skipped wherever they
    /// are. Integers with nulls in the same places are not sorted, because there nulls come first.
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
