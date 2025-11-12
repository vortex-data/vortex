// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod fill_null;
mod is_constant;
mod is_sorted;
mod like;
mod min_max;

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, filter, take,
};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DictArray, DictVTable};

impl TakeKernel for DictVTable {
    fn take(&self, array: &DictArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let codes = take(array.codes(), indices)?;
        // SAFETY: selecting codes doesn't change the invariants of DictArray
        Ok(unsafe { DictArray::new_unchecked(codes, array.values().clone()) }.into_array())
    }
}

register_kernel!(TakeKernelAdapter(DictVTable).lift());

impl FilterKernel for DictVTable {
    fn filter(&self, array: &DictArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let codes = filter(array.codes(), mask)?;

        // SAFETY: filtering codes doesn't change invariants
        unsafe { Ok(DictArray::new_unchecked(codes, array.values().clone()).into_array()) }
    }
}

register_kernel!(FilterKernelAdapter(DictVTable).lift());

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use itertools::Itertools;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray, VarBinArray, VarBinViewArray};
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::compute::{Operator, compare, take};
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, assert_arrays_eq};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability};

    use crate::builders::dict_encode;

    #[test]
    fn canonicalise_nullable_primitive() {
        let values: Vec<Option<i32>> = (0..65)
            .map(|i| match i % 3 {
                0 => Some(42),
                1 => Some(-9),
                2 => None,
                _ => unreachable!(),
            })
            .collect();

        let dict = dict_encode(PrimitiveArray::from_option_iter(values.clone()).as_ref()).unwrap();
        let actual = dict.to_primitive();

        let expected = PrimitiveArray::from_option_iter(values);

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn canonicalise_non_nullable_primitive_32_unique_values() {
        let unique_values: Vec<i32> = (0..32).collect();
        let expected = PrimitiveArray::from_iter((0..1000).map(|i| unique_values[i % 32]));

        let dict = dict_encode(expected.as_ref()).unwrap();
        let actual = dict.to_primitive();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn canonicalise_non_nullable_primitive_100_unique_values() {
        let unique_values: Vec<i32> = (0..100).collect();
        let expected = PrimitiveArray::from_iter((0..1000).map(|i| unique_values[i % 100]));

        let dict = dict_encode(expected.as_ref()).unwrap();
        let actual = dict.to_primitive();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn canonicalise_nullable_varbin() {
        let reference = VarBinViewArray::from_iter(
            vec![Some("a"), Some("b"), None, Some("a"), None, Some("b")],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(reference.len(), 6);
        let dict = dict_encode(reference.as_ref()).unwrap();
        let flattened_dict = dict.to_varbinview();
        assert_eq!(
            flattened_dict
                .with_iterator(|iter| iter
                    .map(|slice| slice.map(|s| s.to_vec()))
                    .collect::<Vec<_>>())
                .unwrap(),
            reference
                .with_iterator(|iter| iter
                    .map(|slice| slice.map(|s| s.to_vec()))
                    .collect::<Vec<_>>())
                .unwrap(),
        );
    }

    fn sliced_dict_array() -> ArrayRef {
        let reference = PrimitiveArray::from_option_iter([
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        dict.slice(1..4)
    }

    #[test]
    fn compare_sliced_dict() {
        use vortex_array::arrays::BoolArray;
        let sliced = sliced_dict_array();
        let compared = compare(&sliced, ConstantArray::new(42, 3).as_ref(), Operator::Eq).unwrap();

        let expected = BoolArray::from_iter([Some(false), None, Some(true)]);
        assert_arrays_eq!(compared, expected.to_array());
    }

    #[test]
    fn test_mask_dict_array() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_mask_conformance(array.as_ref());

        let array = dict_encode(
            PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)]).as_ref(),
        )
        .unwrap();
        test_mask_conformance(array.as_ref());

        let array = dict_encode(
            &VarBinArray::from_iter(
                [
                    Some("hello"),
                    None,
                    Some("hello"),
                    Some("good"),
                    Some("good"),
                ],
                DType::Utf8(Nullability::Nullable),
            )
            .into_array(),
        )
        .unwrap();
        test_mask_conformance(array.as_ref());
    }

    #[test]
    fn test_filter_dict_array() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_filter_conformance(array.as_ref());

        let array = dict_encode(
            PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)]).as_ref(),
        )
        .unwrap();
        test_filter_conformance(array.as_ref());

        let array = dict_encode(
            &VarBinArray::from_iter(
                [
                    Some("hello"),
                    None,
                    Some("hello"),
                    Some("good"),
                    Some("good"),
                ],
                DType::Utf8(Nullability::Nullable),
            )
            .into_array(),
        )
        .unwrap();
        test_filter_conformance(array.as_ref());
    }

    #[test]
    fn test_take_dict() {
        let array = dict_encode(buffer![1, 2].into_array().as_ref()).unwrap();

        assert_eq!(
            take(
                array.as_ref(),
                PrimitiveArray::from_option_iter([Option::<i32>::None]).as_ref()
            )
            .unwrap()
            .dtype(),
            &DType::Primitive(I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_take_dict_conformance() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_take_conformance(array.as_ref());

        let array = dict_encode(
            PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)]).as_ref(),
        )
        .unwrap();
        test_take_conformance(array.as_ref());

        let array = dict_encode(
            &VarBinArray::from_iter(
                [
                    Some("hello"),
                    None,
                    Some("hello"),
                    Some("good"),
                    Some("good"),
                ],
                DType::Utf8(Nullability::Nullable),
            )
            .into_array(),
        )
        .unwrap();
        test_take_conformance(array.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, VarBinArray};
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};

    use crate::DictArray;
    use crate::builders::dict_encode;

    #[rstest]
    // Primitive arrays
    #[case::dict_i32(dict_encode(&buffer![1i32, 2, 3, 2, 1].into_array()).unwrap())]
    #[case::dict_nullable_codes(DictArray::try_new(
        buffer![0u32, 1, 2, 2, 0].into_array(),
        PrimitiveArray::from_option_iter([Some(10), Some(20), None]).into_array(),
    ).unwrap())]
    #[case::dict_nullable_values(dict_encode(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).as_ref()
    ).unwrap())]
    #[case::dict_u64(dict_encode(&buffer![100u64, 200, 100, 300, 200].into_array()).unwrap())]
    // String arrays
    #[case::dict_str(dict_encode(
        &VarBinArray::from_iter(
            ["hello", "world", "hello", "test", "world"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array()
    ).unwrap())]
    #[case::dict_nullable_str(dict_encode(
        &VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("hello"), None],
            DType::Utf8(Nullability::Nullable),
        ).into_array()
    ).unwrap())]
    // Edge cases
    #[case::dict_single(dict_encode(&buffer![42i32].into_array()).unwrap())]
    #[case::dict_all_same(dict_encode(&buffer![5i32, 5, 5, 5, 5].into_array()).unwrap())]
    #[case::dict_large(dict_encode(&PrimitiveArray::from_iter((0..1000).map(|i| i % 10)).into_array()).unwrap())]
    fn test_dict_consistency(#[case] array: DictArray) {
        test_array_consistency(array.as_ref());
    }
}
