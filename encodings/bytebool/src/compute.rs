// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::compute::CastKernel;
use vortex_array::compute::CastKernelAdapter;
use vortex_array::compute::MaskKernel;
use vortex_array::compute::MaskKernelAdapter;
use vortex_array::compute::TakeKernel;
use vortex_array::compute::TakeKernelAdapter;
use vortex_array::register_kernel;
use vortex_array::vtable::ValidityHelper;
use vortex_dtype::DType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::ByteBoolArray;
use super::ByteBoolVTable;

impl CastKernel for ByteBoolVTable {
    fn cast(&self, array: &ByteBoolArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
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

register_kernel!(CastKernelAdapter(ByteBoolVTable).lift());

impl MaskKernel for ByteBoolVTable {
    fn mask(&self, array: &ByteBoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ByteBoolArray::new(array.buffer().clone(), array.validity().mask(mask)).into_array())
    }
}

register_kernel!(MaskKernelAdapter(ByteBoolVTable).lift());

impl TakeKernel for ByteBoolVTable {
    fn take(&self, array: &ByteBoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
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

        Ok(ByteBoolArray::from_vec(taken_bools, validity).into_array())
    }
}

register_kernel!(TakeKernelAdapter(ByteBoolVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;

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

    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    #[test]
    fn test_cast_bytebool_to_nullable() {
        let array = ByteBoolArray::from(vec![true, false, true, false]);
        let casted = cast(array.as_ref(), &DType::Bool(Nullability::Nullable)).unwrap();
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
