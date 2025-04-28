mod compare;
mod to_arrow;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ExtensionEncoding;
use crate::arrays::extension::ExtensionArray;
use crate::compute::{
    CastFn, FilterKernel, FilterKernelAdapter, IsConstantFn, IsConstantOpts, IsSortedFn, MinMaxFn,
    MinMaxResult, ScalarAtFn, SliceFn, SumFn, TakeFn, ToArrowFn, UncompressedSizeFn, filter,
    is_constant_opts, is_sorted, is_strict_sorted, min_max, scalar_at, slice, sum, take,
    uncompressed_size,
};
use crate::variants::ExtensionArrayTrait;
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef, register_kernel};

impl ComputeVTable for ExtensionEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        // It's not possible to cast an extension array to another type.
        // TODO(ngates): we should allow some extension arrays to implement a callback
        //  to support this
        None
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
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

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}

impl FilterKernel for ExtensionEncoding {
    fn filter(&self, array: &ExtensionArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), filter(array.storage(), mask)?)
                .into_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(ExtensionEncoding).lift());

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

impl SumFn<&ExtensionArray> for ExtensionEncoding {
    fn sum(&self, array: &ExtensionArray) -> VortexResult<Scalar> {
        sum(array.storage())
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

impl IsConstantFn<&ExtensionArray> for ExtensionEncoding {
    fn is_constant(
        &self,
        array: &ExtensionArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        is_constant_opts(array.storage(), opts).map(Some)
    }
}

impl UncompressedSizeFn<&ExtensionArray> for ExtensionEncoding {
    fn uncompressed_size(&self, array: &ExtensionArray) -> VortexResult<usize> {
        uncompressed_size(array.storage())
    }
}

impl IsSortedFn<&ExtensionArray> for ExtensionEncoding {
    fn is_sorted(&self, array: &ExtensionArray) -> VortexResult<bool> {
        is_sorted(array.storage())
    }

    fn is_strict_sorted(&self, array: &ExtensionArray) -> VortexResult<bool> {
        is_strict_sorted(array.storage())
    }
}
