mod compare;

use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{
    filter, slice, take, CompareFn, ComputeVTable, FilterFn, FilterMask, SliceFn, TakeFn,
    TakeOptions,
};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DictArray, DictEncoding};

impl ComputeVTable for DictEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<DictArray> for DictEncoding {
    fn scalar_at(&self, array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = scalar_at(array.codes(), index)?.as_ref().try_into()?;
        scalar_at(array.values(), dict_index)
    }
}

impl TakeFn<DictArray> for DictEncoding {
    fn take(
        &self,
        array: &DictArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        // Dict
        //   codes: 0 0 1
        //   dict: a b c d e f g h
        let codes = take(array.codes(), indices, options)?;
        DictArray::try_new(codes, array.values()).map(|a| a.into_array())
    }
}

impl FilterFn<DictArray> for DictEncoding {
    fn filter(&self, array: &DictArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let codes = filter(&array.codes(), mask)?;
        DictArray::try_new(codes, array.values()).map(|a| a.into_array())
    }
}

impl SliceFn<DictArray> for DictEncoding {
    // TODO(robert): Add function to trim the dictionary
    fn slice(&self, array: &DictArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        DictArray::try_new(slice(array.codes(), start, stop)?, array.values())
            .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{ConstantArray, PrimitiveArray, VarBinViewArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{compare, slice, Operator};
    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::{
        dict_encode_primitive, dict_encode_typed_primitive, dict_encode_varbinview, DictArray,
    };

    #[test]
    fn canonicalise_nullable_primitive() {
        let reference = PrimitiveArray::from_nullable_vec(vec![
            Some(42),
            Some(-9),
            None,
            Some(42),
            None,
            Some(-9),
        ]);
        let (codes, values) = dict_encode_typed_primitive::<i32>(&reference);
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let flattened_dict = dict.to_array().into_primitive().unwrap();
        assert_eq!(flattened_dict.buffer(), reference.buffer());
    }

    #[test]
    fn canonicalise_nullable_varbin() {
        let reference = VarBinViewArray::from_iter(
            vec![Some("a"), Some("b"), None, Some("a"), None, Some("b")],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(reference.len(), 6);
        let (codes, values) = dict_encode_varbinview(&reference);
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let flattened_dict = dict.to_array().into_varbinview().unwrap();
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

    #[test]
    fn compare_sliced_dict() {
        let reference = PrimitiveArray::from_nullable_vec(vec![
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let (codes, values) = dict_encode_primitive(&reference);
        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let sliced = slice(dict, 1, 4).unwrap();
        let compared = compare(sliced, ConstantArray::new(42, 3), Operator::Eq).unwrap();
        assert_eq!(
            scalar_at(&compared, 0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(&compared, 1).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(
            scalar_at(compared, 2).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }
}
