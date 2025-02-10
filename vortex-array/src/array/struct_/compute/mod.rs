mod to_arrow;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::struct_::StructArray;
use crate::array::StructEncoding;
use crate::compute::{
    filter, scalar_at, slice, take, FilterFn, MinMaxFn, MinMaxResult, ScalarAtFn, SliceFn, TakeFn,
    ToArrowFn,
};
use crate::variants::StructArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable for StructEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<StructArray> for StructEncoding {
    fn scalar_at(&self, array: &StructArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            array.dtype().clone(),
            array
                .fields()
                .map(|field| scalar_at(&field, index))
                .try_collect()?,
        ))
    }
}

impl TakeFn<StructArray> for StructEncoding {
    fn take(&self, array: &StructArray, indices: &Array) -> VortexResult<Array> {
        StructArray::try_new(
            array.names().clone(),
            array
                .fields()
                .map(|field| take(&field, indices))
                .try_collect()?,
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

impl SliceFn<StructArray> for StructEncoding {
    fn slice(&self, array: &StructArray, start: usize, stop: usize) -> VortexResult<Array> {
        let fields = array
            .fields()
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
    fn filter(&self, array: &StructArray, mask: &Mask) -> VortexResult<Array> {
        let validity = array.validity().filter(mask)?;

        let fields: Vec<Array> = array
            .fields()
            .map(|field| filter(&field, mask))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new(array.names().clone(), fields, length, validity)
            .map(|a| a.into_array())
    }
}

impl MinMaxFn<StructArray> for StructEncoding {
    fn min_max(&self, _array: &StructArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement struct min max
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use crate::array::StructArray;
    use crate::compute::filter;
    use crate::validity::Validity;

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), &Mask::from_iter::<[bool; 0]>([])).unwrap();
        assert_eq!(filtered.len(), 0);
    }
}
