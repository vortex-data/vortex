// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod fill_null;
mod is_constant;
mod is_sorted;
mod like;
mod mask;
pub(crate) mod min_max;
pub(crate) mod rules;
mod slice;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Dict;
use super::DictArray;
use super::TakeExecute;
use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::filter::FilterReduce;

impl TakeExecute for Dict {
    fn take(
        array: &DictArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let codes = array.codes().take(indices.to_array())?;
        // SAFETY: selecting codes doesn't change the invariants of DictArray
        // Preserve all_values_referenced since taking codes doesn't affect which values are referenced
        Ok(Some(unsafe {
            DictArray::new_unchecked(codes, array.values().clone()).into_array()
        }))
    }
}

impl FilterReduce for Dict {
    fn filter(array: &DictArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let codes = array.codes().filter(mask.clone())?;

        // SAFETY: filtering codes doesn't change invariants
        // Preserve all_values_referenced since filtering codes doesn't affect which values are referenced
        Ok(Some(unsafe {
            DictArray::new_unchecked(codes, array.values().clone()).into_array()
        }))
    }
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use itertools::Itertools;
    use vortex_buffer::buffer;

    use crate::ArrayRef;
    use crate::DynArray;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::scalar_fn::fns::operators::Operator;
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

        let dict =
            dict_encode(&PrimitiveArray::from_option_iter(values.clone()).into_array()).unwrap();
        let actual = dict.to_primitive();

        let expected = PrimitiveArray::from_option_iter(values);

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn canonicalise_non_nullable_primitive_32_unique_values() {
        let unique_values: Vec<i32> = (0..32).collect();
        let expected = PrimitiveArray::from_iter((0..1000).map(|i| unique_values[i % 32]));

        let dict = dict_encode(&expected.clone().into_array()).unwrap();
        let actual = dict.to_primitive();

        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn canonicalise_non_nullable_primitive_100_unique_values() {
        let unique_values: Vec<i32> = (0..100).collect();
        let expected = PrimitiveArray::from_iter((0..1000).map(|i| unique_values[i % 100]));

        let dict = dict_encode(&expected.clone().into_array()).unwrap();
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
        let dict = dict_encode(&reference.clone().into_array()).unwrap();
        let flattened_dict = dict.to_varbinview();
        assert_eq!(
            flattened_dict.with_iterator(|iter| iter
                .map(|slice| slice.map(|s| s.to_vec()))
                .collect::<Vec<_>>()),
            reference.with_iterator(|iter| iter
                .map(|slice| slice.map(|s| s.to_vec()))
                .collect::<Vec<_>>()),
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
        let dict = dict_encode(&reference.into_array()).unwrap();
        dict.slice(1..4).unwrap()
    }

    #[test]
    fn compare_sliced_dict() {
        use crate::arrays::BoolArray;
        let sliced = sliced_dict_array();
        let compared = sliced
            .binary(ConstantArray::new(42, 3).into_array(), Operator::Eq)
            .unwrap();

        let expected = BoolArray::from_iter([Some(false), None, Some(true)]);
        assert_arrays_eq!(compared, expected.into_array());
    }

    #[test]
    fn test_mask_dict_array() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_mask_conformance(&array.into_array());

        let array = dict_encode(
            &PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)])
                .into_array(),
        )
        .unwrap();
        test_mask_conformance(&array.into_array());

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
        test_mask_conformance(&array.into_array());
    }

    #[test]
    fn test_filter_dict_array() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_filter_conformance(&array.into_array());

        let array = dict_encode(
            &PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)])
                .into_array(),
        )
        .unwrap();
        test_filter_conformance(&array.into_array());

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
        test_filter_conformance(&array.into_array());
    }

    #[test]
    fn test_take_dict() {
        let array = dict_encode(&buffer![1, 2].into_array()).unwrap();

        assert_eq!(
            array
                .take(PrimitiveArray::from_option_iter([Option::<i32>::None]).into_array())
                .unwrap()
                .dtype(),
            &DType::Primitive(I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_take_dict_conformance() {
        let array = dict_encode(&buffer![2, 0, 2, 0, 10].into_array()).unwrap();
        test_take_conformance(&array.into_array());

        let array = dict_encode(
            &PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)])
                .into_array(),
        )
        .unwrap();
        test_take_conformance(&array.into_array());

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
        test_take_conformance(&array.into_array());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::builders::dict::dict_encode;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    // Primitive arrays
    #[case::dict_i32(dict_encode(&buffer![1i32, 2, 3, 2, 1].into_array()).unwrap())]
    #[case::dict_nullable_codes(DictArray::try_new(
        buffer![0u32, 1, 2, 2, 0].into_array(),
        PrimitiveArray::from_option_iter([Some(10), Some(20), None]).into_array(),
    ).unwrap())]
    #[case::dict_nullable_values(dict_encode(
        &PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).into_array()
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
        test_array_consistency(&array.into_array());
    }
}
