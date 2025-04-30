mod mask;

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ListArray, ListEncoding};
use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, IsSortedFn, MinMaxFn, MinMaxResult,
    ScalarAtFn, SliceFn, UncompressedSizeFn, scalar_at, slice, uncompressed_size,
};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef, register_kernel};

impl ComputeVTable for ListEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&ListArray> for ListEncoding {
    fn scalar_at(&self, array: &ListArray, index: usize) -> VortexResult<Scalar> {
        let elem = array.elements_at(index)?;
        let scalars: Vec<Scalar> = (0..elem.len()).map(|i| scalar_at(&elem, i)).try_collect()?;

        Ok(Scalar::list(
            Arc::new(elem.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        ))
    }
}

impl SliceFn<&ListArray> for ListEncoding {
    fn slice(&self, array: &ListArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ListArray::try_new(
            array.elements().clone(),
            slice(array.offsets(), start, stop + 1)?,
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

impl MinMaxFn<&ListArray> for ListEncoding {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement list min max
        Ok(None)
    }
}

impl IsConstantKernel for ListEncoding {
    fn is_constant(
        &self,
        _array: &ListArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // TODO(adam): Do we want to fallback to arrow here?
        Ok(None)
    }
}

register_kernel!(IsConstantKernelAdapter(ListEncoding).lift());

impl UncompressedSizeFn<&ListArray> for ListEncoding {
    fn uncompressed_size(&self, array: &ListArray) -> VortexResult<usize> {
        let size = uncompressed_size(array.elements())? + uncompressed_size(array.offsets())?;
        Ok(size + array.validity().uncompressed_size())
    }
}

impl IsSortedFn<&ListArray> for ListEncoding {
    fn is_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }

    fn is_strict_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod test {
    use crate::array::Array;
    use crate::arrays::{ListArray, PrimitiveArray};
    use crate::compute::conformance::mask::test_mask;
    use crate::validity::Validity;

    #[test]
    fn test_mask_list() {
        let elements = PrimitiveArray::from_iter(0..35);
        let offsets = PrimitiveArray::from_iter([0, 5, 11, 18, 26, 35]);
        let validity = Validity::AllValid;
        let array =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        test_mask(&array);
    }
}
