// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::DType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::ByteBool;
use super::ByteBoolData;

impl CastReduce for ByteBool {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // ByteBool is essentially a bool array stored as bytes
        // The main difference from BoolArray is the storage format
        // For casting, we can decode to canonical (BoolArray) and let it handle the cast

        // If just changing nullability, we can optimize
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity()
                .clone()
                .cast_nullability(dtype.nullability(), array.len())?;

            return Ok(Some(
                ByteBoolData::new(array.buffer().clone(), new_validity).into_array(),
            ));
        }

        // For other casts, decode to canonical and let BoolArray handle it
        Ok(None)
    }
}

impl MaskReduce for ByteBool {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolData::new(
                array.buffer().clone(),
                array
                    .validity()
                    .clone()
                    .and(Validity::Array(mask.clone()))?,
            )
            .into_array(),
        ))
    }
}

impl TakeExecute for ByteBool {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let bools = array.as_slice();

        // This handles combining validity from both source array and nullable indices
        let validity = array.validity().take(&indices.clone().into_array())?;

        let taken_bools = match_each_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .map(|&idx| {
                    let idx: usize = idx.as_();
                    bools[idx]
                })
                .collect::<Vec<bool>>()
        });

        Ok(Some(
            ByteBoolData::from_vec(taken_bools, validity).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexExpect;

    use super::*;
    use crate::ByteBoolArray;

    fn bb(v: Vec<bool>) -> ByteBoolArray {
        ByteBoolArray::try_from_data(ByteBoolData::from(v))
            .vortex_expect("ByteBoolData is always valid")
    }

    fn bb_opt(v: Vec<Option<bool>>) -> ByteBoolArray {
        ByteBoolArray::try_from_data(ByteBoolData::from(v))
            .vortex_expect("ByteBoolData is always valid")
    }

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = bb_opt(original);

        let sliced_arr = vortex_arr.slice(1..4).unwrap();

        let expected = bb_opt(vec![Some(true), None, Some(false)]);
        assert_arrays_eq!(sliced_arr, expected.into_array());
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = bb(vec![true; 5]);
        let rhs = bb(vec![true; 5]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb(vec![true; 5]);
        assert_arrays_eq!(arr, expected.into_array());
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = bb(vec![false; 5]);
        let rhs = bb(vec![true; 5]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb(vec![false; 5]);
        assert_arrays_eq!(arr, expected.into_array());
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = bb(vec![true; 5]);
        let rhs = bb_opt(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb_opt(vec![Some(true), Some(true), Some(true), Some(false), None]);
        assert_arrays_eq!(arr, expected.into_array());
    }

    #[test]
    fn test_mask_byte_bool() {
        test_mask_conformance(&bb(vec![true, false, true, true, false]).into_array());
        test_mask_conformance(
            &bb_opt(vec![Some(true), Some(true), None, Some(false), None]).into_array(),
        );
    }

    #[test]
    fn test_filter_byte_bool() {
        test_filter_conformance(&bb(vec![true, false, true, true, false]).into_array());
        test_filter_conformance(
            &bb_opt(vec![Some(true), Some(true), None, Some(false), None]).into_array(),
        );
    }

    #[rstest]
    #[case(bb(vec![true, false, true, true, false]))]
    #[case(bb_opt(vec![Some(true), Some(true), None, Some(false), None]))]
    #[case(bb(vec![true, false]))]
    #[case(bb(vec![true]))]
    fn test_take_byte_bool_conformance(#[case] array: ByteBoolArray) {
        test_take_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_bytebool_to_nullable() {
        let array = bb(vec![true, false, true, false]);
        let casted = array
            .into_array()
            .cast(DType::Bool(Nullability::Nullable))
            .unwrap();
        assert_eq!(casted.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(casted.len(), 4);
    }

    #[rstest]
    #[case(bb(vec![true, false, true, true, false]))]
    #[case(bb_opt(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(bb(vec![false]))]
    #[case(bb(vec![true]))]
    #[case(bb_opt(vec![Some(true), None]))]
    fn test_cast_bytebool_conformance(#[case] array: ByteBoolArray) {
        test_cast_conformance(&array.into_array());
    }

    #[rstest]
    #[case::non_nullable(bb(vec![true, false, true, true, false]))]
    #[case::nullable(bb_opt(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case::all_true(bb(vec![true, true, true, true]))]
    #[case::all_false(bb(vec![false, false, false, false]))]
    #[case::single_true(bb(vec![true]))]
    #[case::single_false(bb(vec![false]))]
    #[case::single_null(bb_opt(vec![None]))]
    #[case::mixed_with_nulls(bb_opt(vec![Some(true), None, Some(false), None, Some(true)]))]
    fn test_bytebool_consistency(#[case] array: ByteBoolArray) {
        test_array_consistency(&array.into_array());
    }
}
