mod compare;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::extension::ExtensionArray;
use crate::array::ExtensionEncoding;
use crate::compute::unary::{scalar_at, CastFn, ScalarAtFn};
use crate::compute::{slice, take, CompareFn, ComputeVTable, SliceFn, TakeFn, TakeOptions};
use crate::variants::ExtensionArrayTrait;
use crate::{ArrayData, IntoArrayData};

impl ComputeVTable for ExtensionEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
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

impl ScalarAtFn<ExtensionArray> for ExtensionEncoding {
    fn scalar_at(&self, array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            array.ext_dtype().clone(),
            scalar_at(array.storage(), index)?,
        ))
    }
}

impl SliceFn<ExtensionArray> for ExtensionEncoding {
    fn slice(&self, array: &ExtensionArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            slice(array.storage(), start, stop)?,
        )
        .into_array())
    }
}

impl TakeFn<ExtensionArray> for ExtensionEncoding {
    fn take(
        &self,
        array: &ExtensionArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            take(array.storage(), indices, options)?,
        )
        .into_array())
    }
}
