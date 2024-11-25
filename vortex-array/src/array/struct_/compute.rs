use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::struct_::StructArray;
use crate::array::StructEncoding;
use crate::compute::unary::{scalar_at, ScalarAtFn};
use crate::compute::{
    filter, slice, take, ComputeVTable, FilterFn, FilterMask, SliceFn, TakeFn, TakeOptions,
};
use crate::variants::StructArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ComputeVTable for StructEncoding {
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

impl ScalarAtFn<StructArray> for StructEncoding {
    fn scalar_at(&self, array: &StructArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            array.dtype().clone(),
            array
                .children()
                .map(|field| scalar_at(&field, index))
                .try_collect()?,
        ))
    }
}

impl TakeFn<StructArray> for StructEncoding {
    fn take(
        &self,
        array: &StructArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        StructArray::try_new(
            array.names().clone(),
            array
                .children()
                .map(|field| take(&field, indices, options))
                .try_collect()?,
            indices.len(),
            array.validity().take(indices, options)?,
        )
        .map(|a| a.into_array())
    }
}

impl SliceFn<StructArray> for StructEncoding {
    fn slice(&self, array: &StructArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let fields = array
            .children()
            .map(|field| slice(&field, start, stop))
            .try_collect()?;
        StructArray::try_new(
            array.names().clone(),
            fields,
            stop - start,
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn<StructArray> for StructEncoding {
    fn filter(&self, array: &StructArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(&mask)?;

        let fields: Vec<ArrayData> = array
            .children()
            .map(|field| filter(&field, mask.clone()))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new(array.names().clone(), fields, length, validity)
            .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::StructArray;
    use crate::compute::{filter, FilterMask};
    use crate::validity::Validity;

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), FilterMask::from_iter(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), FilterMask::from_iter::<[bool; 0]>([])).unwrap();
        assert_eq!(filtered.len(), 0);
    }
}
