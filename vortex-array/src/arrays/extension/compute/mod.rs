mod compare;
mod to_arrow;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::extension::ExtensionArray;
use crate::arrays::ExtensionEncoding;
use crate::compute::{
    filter, min_max, scalar_at, slice, take, CastFn, CompareFn, FilterFn, MinMaxFn, MinMaxResult,
    ScalarAtFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::variants::ExtensionArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef, IntoArray};

impl ComputeVTable for ExtensionEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
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

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }
}

impl FilterFn<&ExtensionArray> for ExtensionEncoding {
    fn filter(&self, array: &ExtensionArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), filter(array.storage(), mask)?)
                .into_array(),
        )
    }
}

impl ScalarAtFn<&ExtensionArray> for ExtensionEncoding {
    fn scalar_at(&self, array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            array.ext_dtype().clone(),
            scalar_at(array.storage(), index)?,
        ))
    }
}

impl SliceFn<&ExtensionArray> for ExtensionEncoding {
    fn slice(&self, array: &ExtensionArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            slice(array.storage(), start, stop)?,
        )
        .into_array())
    }
}

impl TakeFn<&ExtensionArray> for ExtensionEncoding {
    fn take(&self, array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), take(array.storage(), indices)?)
                .into_array(),
        )
    }
}

impl MinMaxFn<&ExtensionArray> for ExtensionEncoding {
    fn min_max(&self, array: &ExtensionArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(
            min_max(array.storage())?.map(|MinMaxResult { min, max }| MinMaxResult {
                min: Scalar::extension(array.ext_dtype().clone(), min),
                max: Scalar::extension(array.ext_dtype().clone(), max),
            }),
        )
    }
}
