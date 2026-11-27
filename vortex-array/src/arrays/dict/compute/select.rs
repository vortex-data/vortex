// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::cast::casts_non_null_values_losslessly;
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
use crate::scalar_fn::builtin::cast::Cast;

/// Returns the rows of `array` that `codes` select.
///
/// Fallible scalar functions in the values run only on the selected rows, so that unreferenced
/// values cannot fail.
pub(super) fn select_codes(array: ArrayView<'_, Dict>, codes: ArrayRef) -> VortexResult<ArrayRef> {
    let values = array.values();

    if !has_fallible_scalar_fn(values) {
        // SAFETY: Selection preserves the code type and bounds. `new_unchecked` resets
        // `all_values_referenced`.
        let selected = unsafe { DictArray::new_unchecked(codes, values.clone()) };
        return Ok(selected.into_array());
    }

    let selected = take_through_scalar_fns(values, &codes)?;

    if codes.dtype().is_nullable() {
        let valid_codes = codes.is_not_null()?;
        return selected.mask(valid_codes);
    }

    Ok(selected)
}

/// Returns whether the scalar function tree of `array` contains a fallible function.
fn has_fallible_scalar_fn(array: &ArrayRef) -> bool {
    let Some(scalar_fn) = array.as_opt::<ScalarFn>() else {
        return false;
    };

    !is_infallible(&scalar_fn) || scalar_fn.iter_children().any(has_fallible_scalar_fn)
}

/// Like the signature's `is_infallible`, but also accepts lossless casts.
fn is_infallible(array: &ArrayView<'_, ScalarFn>) -> bool {
    let scalar_fn = array.scalar_fn();
    if scalar_fn.signature().is_infallible() {
        return true;
    }

    if !scalar_fn.is::<Cast>() {
        return false;
    }

    let from = array.get_child(0).dtype();
    let to = array.dtype();
    let null_safe = to.is_nullable() || !from.is_nullable();

    null_safe && casts_non_null_values_losslessly(from, to)
}

/// Applies the scalar functions of `array` to the rows of their inputs that `codes` select.
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
    use vortex_error::VortexExpect;
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
    use crate::arrays::ScalarFn;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::filter::FilterReduce;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
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

    #[test]
    fn selection_keeps_dictionary_with_lossless_cast_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 1].into_array(),
            buffer![10u8, 20].into_array(),
        )?
        .into_array();

        let cast = dict
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))?
            .optimize_ctx(&SESSION)?;
        assert!(cast.as_::<Dict>().values().is::<ScalarFn>());

        // The optimizer would rebuild the dictionary, so check the kernel output.
        let mask = Mask::from_iter([true, true, false, true]);
        let selected = <Dict as FilterReduce>::filter(cast.as_::<Dict>(), &mask)?
            .vortex_expect("Dict FilterReduce always returns Some");

        assert!(selected.is::<Dict>());
        assert_arrays_eq!(
            selected,
            PrimitiveArray::from_iter([10i64, 20, 20]),
            &mut ctx
        );
        Ok(())
    }

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
