use vortex_array::compute::unary::{scalar_at, scalar_at_unchecked, ScalarAtFn};
use vortex_array::compute::{
    compare, filter, slice, take, ArrayCompute, FilterFn, FilterMask, MaybeCompareFn, Operator,
    SliceFn, TakeFn, TakeOptions,
};
use vortex_array::stats::{ArrayStatistics, Stat};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::DictArray;

impl ArrayCompute for DictArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl MaybeCompareFn for DictArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if other
            .statistics()
            .get_as::<bool>(Stat::IsConstant)
            .unwrap_or_default()
        {
            return Some(
                // Ensure the other is the same length as the dictionary
                slice(other, 0, self.values().len())
                    .and_then(|other| compare(self.values(), other, operator))
                    .and_then(|values| Self::try_new(self.codes().clone(), values))
                    .map(|a| a.into_array()),
            );
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        None
    }
}

impl ScalarAtFn for DictArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = scalar_at(self.codes(), index)?.as_ref().try_into()?;
        Ok(scalar_at_unchecked(self.values(), dict_index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        let dict_index: usize = scalar_at_unchecked(self.codes(), index)
            .as_ref()
            .try_into()
            .vortex_expect("Invalid dict index");

        scalar_at_unchecked(self.values(), dict_index)
    }
}

impl TakeFn for DictArray {
    fn take(&self, indices: &ArrayData, options: TakeOptions) -> VortexResult<ArrayData> {
        // Dict
        //   codes: 0 0 1
        //   dict: a b c d e f g h
        let codes = take(self.codes(), indices, options)?;
        Self::try_new(codes, self.values()).map(|a| a.into_array())
    }
}

impl FilterFn for DictArray {
    fn filter(&self, mask: FilterMask) -> VortexResult<ArrayData> {
        let codes = filter(&self.codes(), mask)?;
        Self::try_new(codes, self.values()).map(|a| a.into_array())
    }
}

impl SliceFn for DictArray {
    // TODO(robert): Add function to trim the dictionary
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Self::try_new(slice(self.codes(), start, stop)?, self.values()).map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{PrimitiveArray, VarBinViewArray};
    use vortex_array::{IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_dtype::{DType, Nullability};

    use crate::{dict_encode_typed_primitive, dict_encode_varbinview, DictArray};

    #[test]
    fn flatten_nullable_primitive() {
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
    fn flatten_nullable_varbin() {
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
}
