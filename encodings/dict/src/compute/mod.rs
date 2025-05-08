mod binary_numeric;
mod compare;
mod fill_null;
mod is_constant;
mod is_sorted;
mod like;
mod min_max;

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, filter, take,
};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DictArray, DictEncoding};

impl TakeKernel for DictEncoding {
    fn take(&self, array: &DictArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let codes = take(array.codes(), indices)?;
        DictArray::try_new(codes, array.values().clone()).map(|a| a.into_array())
    }
}

register_kernel!(TakeKernelAdapter(DictEncoding).lift());

impl FilterKernel for DictEncoding {
    fn filter(&self, array: &DictArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let codes = filter(array.codes(), mask)?;
        DictArray::try_new(codes, array.values().clone()).map(|a| a.into_array())
    }
}

register_kernel!(FilterKernelAdapter(DictEncoding).lift());

#[cfg(test)]
mod test {
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray, VarBinArray, VarBinViewArray};
    use vortex_array::compute::conformance::mask::test_mask;
    use vortex_array::compute::{Operator, compare};
    use vortex_array::{Array, ArrayRef, ToCanonical};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

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

        let dict = dict_encode(&PrimitiveArray::from_option_iter(values.clone())).unwrap();
        let actual = dict.to_primitive().unwrap();

        let expected: Vec<i32> = (0..65)
            .map(|i| match i % 3 {
                // Compressor puts 0 as a code for invalid values which we end up using in take
                // thus invalid values on decompression turn into whatever is at 0th position in dictionary
                0 | 2 => 42,
                1 => -9,
                _ => unreachable!(),
            })
            .collect();

        assert_eq!(actual.as_slice::<i32>(), expected.as_slice());

        let expected_valid_count = values.iter().filter(|x| x.is_some()).count();
        assert_eq!(
            actual.validity_mask().unwrap().true_count(),
            expected_valid_count
        );
    }

    #[test]
    fn canonicalise_non_nullable_primitive_32_unique_values() {
        let unique_values: Vec<i32> = (0..32).collect();
        let expected: Vec<i32> = (0..1000).map(|i| unique_values[i % 32]).collect();

        let dict = dict_encode(&PrimitiveArray::from_iter(expected.iter().copied())).unwrap();
        let actual = dict.to_primitive().unwrap();

        assert_eq!(actual.as_slice::<i32>(), expected.as_slice());
    }

    #[test]
    fn canonicalise_non_nullable_primitive_100_unique_values() {
        let unique_values: Vec<i32> = (0..100).collect();
        let expected: Vec<i32> = (0..1000).map(|i| unique_values[i % 100]).collect();

        let dict = dict_encode(&PrimitiveArray::from_iter(expected.iter().copied())).unwrap();
        let actual = dict.to_primitive().unwrap();

        assert_eq!(actual.as_slice::<i32>(), expected.as_slice());
    }

    #[test]
    fn canonicalise_nullable_varbin() {
        let reference = VarBinViewArray::from_iter(
            vec![Some("a"), Some("b"), None, Some("a"), None, Some("b")],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(reference.len(), 6);
        let dict = dict_encode(&reference).unwrap();
        let flattened_dict = dict.to_varbinview().unwrap();
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
        let dict = dict_encode(&reference).unwrap();
        dict.slice(1, 4).unwrap()
    }

    #[test]
    fn compare_sliced_dict() {
        let sliced = sliced_dict_array();
        let compared = compare(&sliced, &ConstantArray::new(42, 3), Operator::Eq).unwrap();

        assert_eq!(
            compared.scalar_at(0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            compared.scalar_at(1).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(
            compared.scalar_at(2).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    fn test_mask_dict_array() {
        let array = dict_encode(&PrimitiveArray::from_iter([2, 0, 2, 0, 10]).into_array()).unwrap();
        test_mask(&array);

        let array = dict_encode(
            &PrimitiveArray::from_option_iter([Some(2), None, Some(2), Some(0), Some(10)])
                .into_array(),
        )
        .unwrap();
        test_mask(&array);

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
        test_mask(&array);
    }
}
