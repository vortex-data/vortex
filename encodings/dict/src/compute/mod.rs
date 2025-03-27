mod binary_numeric;
mod compare;
mod is_constant;
mod is_sorted;
mod like;
mod min_max;
mod optimize;

use vortex_array::compute::{
    BinaryNumericFn, CompareFn, FilterKernel, FilterKernelAdapter, IsConstantFn, IsSortedFn,
    KernelRef, LikeFn, MinMaxFn, OptimizeFn, ScalarAtFn, SliceFn, TakeFn, filter, scalar_at, slice,
    take,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{DictArray, DictEncoding};

impl ArrayComputeImpl for DictArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(DictEncoding).some();
}

impl ComputeVTable for DictEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn like_fn(&self) -> Option<&dyn LikeFn<&dyn Array>> {
        Some(self)
    }

    fn optimize_fn(&self) -> Option<&dyn OptimizeFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&DictArray> for DictEncoding {
    fn scalar_at(&self, array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = scalar_at(array.codes(), index)?.as_ref().try_into()?;
        scalar_at(array.values(), dict_index)
    }
}

impl TakeFn<&DictArray> for DictEncoding {
    fn take(&self, array: &DictArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let codes = take(array.codes(), indices)?;
        DictArray::try_new(codes, array.values().clone()).map(|a| a.into_array())
    }
}

impl FilterKernel for DictEncoding {
    fn filter(&self, array: &DictArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let codes = filter(array.codes(), mask)?;
        DictArray::try_new(codes, array.values().clone()).map(|a| a.into_array())
    }
}

impl SliceFn<&DictArray> for DictEncoding {
    // TODO(robert): Add function to trim the dictionary
    fn slice(&self, array: &DictArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DictArray::try_new(slice(array.codes(), start, stop)?, array.values().clone())
            .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray, VarBinArray, VarBinViewArray};
    use vortex_array::compute::conformance::mask::test_mask;
    use vortex_array::compute::{Operator, compare, scalar_at, slice};
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
        slice(&dict, 1, 4).unwrap()
    }

    #[test]
    fn compare_sliced_dict() {
        let sliced = sliced_dict_array();
        let compared = compare(&sliced, &ConstantArray::new(42, 3), Operator::Eq).unwrap();

        assert_eq!(
            scalar_at(&compared, 0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(&compared, 1).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(
            scalar_at(&compared, 2).unwrap(),
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
