mod compare;
mod to_arrow;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::extension::ExtensionArray;
use crate::array::ExtensionEncoding;
use crate::compute::{
    scalar_at, slice, take, CastFn, CompareFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::variants::ExtensionArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable for ExtensionEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
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
}

impl ScalarAtFn<ExtensionArray> for ExtensionEncoding {
    fn scalar_at(&self, array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            array.ext_dtype().clone(),
            scalar_at(array.storage(), index)?,
        ))
    }
}

impl SliceFn<ExtensionArray> for ExtensionEncoding {
    fn slice(&self, array: &ExtensionArray, start: usize, stop: usize) -> VortexResult<Array> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            slice(array.storage(), start, stop)?,
        )
        .into_array())
    }
}

impl TakeFn<ExtensionArray> for ExtensionEncoding {
    fn take(&self, array: &ExtensionArray, indices: &Array) -> VortexResult<Array> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), take(array.storage(), indices)?)
                .into_array(),
        )
    }
}
