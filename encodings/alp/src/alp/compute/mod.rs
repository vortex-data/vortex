mod compare;

use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{
    filter, slice, take, CompareFn, ComputeVTable, FilterFn, FilterMask, SliceFn, TakeFn,
    TakeOptions,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{match_each_alp_float_ptype, ALPArray, ALPEncoding, ALPFloat};

impl ComputeVTable for ALPEncoding {
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

impl ScalarAtFn<ALPArray> for ALPEncoding {
    fn scalar_at(&self, array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = array.patches() {
            if patches.with_dyn(|a| a.is_valid(index)) {
                // We need to make sure the value is actually in the patches array
                return scalar_at(&patches, index);
            }
        }

        let encoded_val = scalar_at(array.encoded(), index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                array.exponents(),
            ), array.dtype().nullability())
        }))
    }
}

impl TakeFn<ALPArray> for ALPEncoding {
    fn take(
        &self,
        array: &ALPArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        // TODO(ngates): wrap up indices in an array that caches decompression?
        Ok(ALPArray::try_new(
            take(array.encoded(), indices, options)?,
            array.exponents(),
            array
                .patches()
                .map(|p| take(&p, indices, options))
                .transpose()?,
        )?
        .into_array())
    }
}

impl SliceFn<ALPArray> for ALPEncoding {
    fn slice(&self, array: &ALPArray, start: usize, end: usize) -> VortexResult<ArrayData> {
        Ok(ALPArray::try_new(
            slice(array.encoded(), start, end)?,
            array.exponents(),
            array.patches().map(|p| slice(&p, start, end)).transpose()?,
        )?
        .into_array())
    }
}

impl FilterFn<ALPArray> for ALPEncoding {
    fn filter(&self, array: &ALPArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(ALPArray::try_new(
            filter(&array.encoded(), mask.clone())?,
            array.exponents(),
            array.patches().map(|p| filter(&p, mask)).transpose()?,
        )?
        .into_array())
    }
}
