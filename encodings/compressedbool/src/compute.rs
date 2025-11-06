// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{
    CastKernel, CastKernelAdapter, MaskKernel, MaskKernelAdapter, TakeKernel, TakeKernelAdapter,
};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::{CompressedBoolArray, CompressedBoolVTable};

impl CastKernel for CompressedBoolVTable {
    fn cast(&self, array: &CompressedBoolArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // CompressedBool is essentially a bool array stored as bytes
        // The main difference from BoolArray is the storage format
        // For casting, we can decode to canonical (BoolArray) and let it handle the cast

        // If just changing nullability, we can optimize
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity()
                .clone()
                .cast_nullability(dtype.nullability(), array.len())?;

            return Ok(Some(
                CompressedBoolArray::try_new(
                    array.compressed_buffer().clone(),
                    new_validity,
                    array.bit_offset(),
                    array.len(),
                )?
                .into_array(),
            ));
        }

        // For other casts, decode to canonical and let BoolArray handle it
        Ok(None)
    }
}

register_kernel!(CastKernelAdapter(CompressedBoolVTable).lift());

impl MaskKernel for CompressedBoolVTable {
    fn mask(&self, array: &CompressedBoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(CompressedBoolArray::try_new(
            array.compressed_buffer().clone(),
            array.validity().mask(mask),
            array.bit_offset(),
            array.len(),
        )?
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(CompressedBoolVTable).lift());

impl TakeKernel for CompressedBoolVTable {
    fn take(&self, array: &CompressedBoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        todo!()
        // let indices = indices.to_primitive();
        // let bools = array.as_slice();

        // // This handles combining validity from both source array and nullable indices
        // let validity = array.validity().take(indices.as_ref())?;

        // let taken_bools = match_each_integer_ptype!(indices.ptype(), |I| {
        //     indices
        //         .as_slice::<I>()
        //         .iter()
        //         .map(|&idx| {
        //             let idx: usize = idx.as_();
        //             bools[idx]
        //         })
        //         .collect::<Vec<bool>>()
        // });

        // Ok(CompressedBoolArray::from_vec(taken_bools, validity).into_array())
    }
}

register_kernel!(TakeKernelAdapter(CompressedBoolVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::{Operator, compare};

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = CompressedBoolArray::from(original);

        let sliced_arr = vortex_arr.slice(1..4);

        let expected = CompressedBoolArray::from(vec![Some(true), None, Some(false)]);
        assert_arrays_eq!(sliced_arr, expected.to_array());
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = CompressedBoolArray::from(vec![true; 5]);
        let rhs = CompressedBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected = CompressedBoolArray::from(vec![true; 5]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = CompressedBoolArray::from(vec![false; 5]);
        let rhs = CompressedBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected = CompressedBoolArray::from(vec![false; 5]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = CompressedBoolArray::from(vec![true; 5]);
        let rhs =
            CompressedBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        let expected =
            CompressedBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);
        assert_arrays_eq!(arr, expected.to_array());
    }

    #[test]
    fn test_mask_byte_bool() {
        test_mask_conformance(
            CompressedBoolArray::from(vec![true, false, true, true, false]).as_ref(),
        );
        test_mask_conformance(
            CompressedBoolArray::from(vec![Some(true), Some(true), None, Some(false), None])
                .as_ref(),
        );
    }

    #[test]
    fn test_filter_byte_bool() {
        test_filter_conformance(
            CompressedBoolArray::from(vec![true, false, true, true, false]).as_ref(),
        );
        test_filter_conformance(
            CompressedBoolArray::from(vec![Some(true), Some(true), None, Some(false), None])
                .as_ref(),
        );
    }

    #[rstest]
    #[case(CompressedBoolArray::from(vec![true, false, true, true, false]))]
    #[case(CompressedBoolArray::from(vec![Some(true), Some(true), None, Some(false), None]))]
    #[case(CompressedBoolArray::from(vec![true, false]))]
    #[case(CompressedBoolArray::from(vec![true]))]
    fn test_take_byte_bool_conformance(#[case] array: CompressedBoolArray) {
        test_take_conformance(array.as_ref());
    }

    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_dtype::{DType, Nullability};

    #[test]
    fn test_cast_bytebool_to_nullable() {
        let array = CompressedBoolArray::from(vec![true, false, true, false]);
        let casted = cast(array.as_ref(), &DType::Bool(Nullability::Nullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(casted.len(), 4);
    }

    #[rstest]
    #[case(CompressedBoolArray::from(vec![true, false, true, true, false]))]
    #[case(CompressedBoolArray::from(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(CompressedBoolArray::from(vec![false]))]
    #[case(CompressedBoolArray::from(vec![true]))]
    #[case(CompressedBoolArray::from(vec![Some(true), None]))]
    fn test_cast_bytebool_conformance(#[case] array: CompressedBoolArray) {
        test_cast_conformance(array.as_ref());
    }

    #[rstest]
    #[case::non_nullable(CompressedBoolArray::from(vec![true, false, true, true, false]))]
    #[case::nullable(CompressedBoolArray::from(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case::all_true(CompressedBoolArray::from(vec![true, true, true, true]))]
    #[case::all_false(CompressedBoolArray::from(vec![false, false, false, false]))]
    #[case::single_true(CompressedBoolArray::from(vec![true]))]
    #[case::single_false(CompressedBoolArray::from(vec![false]))]
    #[case::single_null(CompressedBoolArray::from(vec![None]))]
    #[case::mixed_with_nulls(CompressedBoolArray::from(vec![Some(true), None, Some(false), None, Some(true)]))]
    fn test_bytebool_consistency(#[case] array: CompressedBoolArray) {
        test_array_consistency(array.as_ref());
    }
}
