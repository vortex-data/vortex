// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::TakeExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::CastReduce;
use vortex_array::expr::MaskReduce;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use super::ByteBoolArray;
use super::ByteBoolVTable;

impl CastReduce for ByteBoolVTable {
    fn cast(array: &ByteBoolArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
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
                ByteBoolArray::new(array.buffer().clone(), new_validity).into_array(),
            ));
        }

        // For other casts, decode to canonical and let BoolArray handle it
        Ok(None)
    }
}

impl MaskReduce for ByteBoolVTable {
    fn mask(array: &ByteBoolArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBoolArray::new(
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

impl TakeExecute for ByteBoolVTable {
    fn take(
        array: &ByteBoolArray,
        indices: &dyn Array,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.to_primitive();
        let bools = array.as_slice();

        // This handles combining validity from both source array and nullable indices
        let validity = array.validity().take(indices.as_ref())?;

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
            ByteBoolArray::from_vec(taken_bools, validity).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = ByteBoolArray::from(original);

        let sliced_arr = vortex_arr.slice(1..4).unwrap();

        let expected = ByteBoolArray::from(vec![Some(true), None, Some(false)]);
        assert_arrays_eq!(sliced_arr, expected.to_array());
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected = ByteBoolArray::from(vec![true; 5]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = ByteBoolArray::from(vec![false; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected = ByteBoolArray::from(vec![false; 5]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected =
            ByteBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_mask_byte_bool() {
        test_mask_conformance(ByteBoolArray::from(vec![true, false, true, true, false]).as_ref());
        test_mask_conformance(
            ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]).as_ref(),
        );
    }

    #[test]
    fn test_filter_byte_bool() {
        test_filter_conformance(ByteBoolArray::from(vec![true, false, true, true, false]).as_ref());
        test_filter_conformance(
            ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]).as_ref(),
        );
    }

    #[rstest]
    #[case(ByteBoolArray::from(vec![true, false, true, true, false]))]
    #[case(ByteBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]))]
    #[case(ByteBoolArray::from(vec![true, false]))]
    #[case(ByteBoolArray::from(vec![true]))]
    fn test_take_byte_bool_conformance(#[case] array: ByteBoolArray) {
        test_take_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_bytebool_to_nullable() {
        let array = ByteBoolArray::from(vec![true, false, true, false]);
        let casted = array
            .to_array()
            .cast(DType::Bool(Nullability::Nullable))
            .unwrap();
        assert_eq!(casted.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(casted.len(), 4);
    }

    #[rstest]
    #[case(ByteBoolArray::from(vec![true, false, true, true, false]))]
    #[case(ByteBoolArray::from(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(ByteBoolArray::from(vec![false]))]
    #[case(ByteBoolArray::from(vec![true]))]
    #[case(ByteBoolArray::from(vec![Some(true), None]))]
    fn test_cast_bytebool_conformance(#[case] array: ByteBoolArray) {
        test_cast_conformance(array.as_ref());
    }

    #[rstest]
    #[case::non_nullable(ByteBoolArray::from(vec![true, false, true, true, false]))]
    #[case::nullable(ByteBoolArray::from(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case::all_true(ByteBoolArray::from(vec![true, true, true, true]))]
    #[case::all_false(ByteBoolArray::from(vec![false, false, false, false]))]
    #[case::single_true(ByteBoolArray::from(vec![true]))]
    #[case::single_false(ByteBoolArray::from(vec![false]))]
    #[case::single_null(ByteBoolArray::from(vec![None]))]
    #[case::mixed_with_nulls(ByteBoolArray::from(vec![Some(true), None, Some(false), None, Some(true)]))]
    fn test_bytebool_consistency(#[case] array: ByteBoolArray) {
        test_array_consistency(array.as_ref());
    }
}
