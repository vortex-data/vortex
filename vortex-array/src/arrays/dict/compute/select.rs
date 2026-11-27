// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::builtins::ArrayBuiltins;

/// Returns the rows of `array` that `codes` select from its values.
///
/// A selection can remove the last reference to a value. A fallible scalar function that a
/// dictionary with all values referenced pushed into its values would then also run on values
/// that no selected row references, and it can fail where row-wise evaluation does not. For
/// such values, this applies the scalar functions to the selected rows of their inputs, so that
/// selection precedes evaluation.
pub(super) fn select_codes(array: ArrayView<'_, Dict>, codes: ArrayRef) -> VortexResult<ArrayRef> {
    let values = array.values();

    if !has_fallible_scalar_fn(values) {
        // SAFETY: Selection preserves the integer code type and non-null code bounds, so every
        // code indexes the unchanged values. `new_unchecked` resets `all_values_referenced`
        // because selection can remove the last reference to a value.
        let selected = unsafe { DictArray::new_unchecked(codes, values.clone()) };
        return Ok(selected.into_array());
    }

    let selected = take_through_scalar_fns(values, &codes)?;

    // A null code gives a null row, whatever the scalar function returns for null inputs.
    if codes.dtype().is_nullable() {
        let valid_codes = codes.is_not_null()?;
        return selected.mask(valid_codes);
    }

    Ok(selected)
}

/// Returns whether `array` is a scalar function tree that contains a fallible function.
fn has_fallible_scalar_fn(array: &ArrayRef) -> bool {
    let Some(scalar_fn) = array.as_opt::<ScalarFn>() else {
        return false;
    };

    !scalar_fn.scalar_fn().signature().is_infallible()
        || scalar_fn.iter_children().any(has_fallible_scalar_fn)
}

/// Takes `codes` from the inputs of the scalar functions of `array`, and applies the functions
/// to the result.
fn take_through_scalar_fns(array: &ArrayRef, codes: &ArrayRef) -> VortexResult<ArrayRef> {
    let Some(scalar_fn) = array.as_opt::<ScalarFn>() else {
        return array.take(codes.clone());
    };

    let children = scalar_fn
        .iter_children()
        .map(|child| match child.as_opt::<Constant>() {
            Some(constant) => {
                Ok(ConstantArray::new(constant.scalar().clone(), codes.len()).into_array())
            }
            None => take_through_scalar_fns(child, codes),
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let selected =
        ScalarFnArray::try_new_with_len(scalar_fn.scalar_fn().clone(), children, codes.len())?;

    Ok(selected.into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::NumericOperator;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Returns `12 / dict` with the division pushed into the values `[0, 1, 2]`.
    fn division_over_values(codes: ArrayRef) -> VortexResult<ArrayRef> {
        // SAFETY: The codes reference every value.
        let dict = unsafe {
            DictArray::try_new(codes, buffer![0i32, 1, 2].into_array())?
                .set_all_values_referenced(true)
        }
        .into_array();

        let division = ConstantArray::new(12i32, dict.len())
            .into_array()
            .numeric(dict, NumericOperator::Div)?
            .optimize_ctx(&SESSION)?;
        assert!(division.is::<Dict>());

        Ok(division)
    }

    /// A selection of rows that do not reference the value zero must not divide by it.
    #[rstest]
    #[case::filter(
        |array: &ArrayRef| array.filter(Mask::from_iter([false, false, true, true, false, true])),
        PrimitiveArray::from_iter([12i32, 6, 12]),
    )]
    #[case::take(
        |array: &ArrayRef| array.take(buffer![2u32, 3, 5].into_array()),
        PrimitiveArray::from_iter([12i32, 6, 12]),
    )]
    #[case::slice(
        |array: &ArrayRef| array.slice(2..4),
        PrimitiveArray::from_iter([12i32, 6]),
    )]
    #[case::take_null_index(
        |array: &ArrayRef| array.take(PrimitiveArray::from_option_iter([Some(2u32), None]).into_array()),
        PrimitiveArray::from_option_iter([Some(12i32), None]),
    )]
    fn selection_skips_fallible_values_of_unselected_rows(
        #[case] select: fn(&ArrayRef) -> VortexResult<ArrayRef>,
        #[case] expected: PrimitiveArray,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let division = division_over_values(buffer![0u8, 0, 1, 2, 0, 1].into_array())?;

        let selected = select(&division)?.optimize_ctx(&SESSION)?;

        assert_arrays_eq!(selected, expected, &mut ctx);
        Ok(())
    }

    /// A null code selects a null row, even though the selected value would not be null.
    #[test]
    fn null_codes_select_null_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = PrimitiveArray::from_option_iter([Some(1u8), None, Some(2), Some(0)]);
        let division = division_over_values(codes.into_array())?;

        let selected = division
            .filter(Mask::from_iter([true, true, true, false]))?
            .optimize_ctx(&SESSION)?;

        assert_arrays_eq!(
            selected,
            PrimitiveArray::from_option_iter([Some(12i32), None, Some(6)]),
            &mut ctx
        );
        Ok(())
    }
}
