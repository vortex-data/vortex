mod mask;

use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListEncoding};
use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, IsSortedKernel,
    IsSortedKernelAdapter, MinMaxKernel, MinMaxKernelAdapter, MinMaxResult, UncompressedSizeFn,
    uncompressed_size,
};
use crate::vtable::ComputeVTable;
use crate::{Array, register_kernel};

impl ComputeVTable for ListEncoding {
    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
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

impl MinMaxKernel for ListEncoding {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement list min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(ListEncoding).lift());

impl UncompressedSizeFn<&ListArray> for ListEncoding {
    fn uncompressed_size(&self, array: &ListArray) -> VortexResult<usize> {
        let size = uncompressed_size(array.elements())? + uncompressed_size(array.offsets())?;
        Ok(size + array.validity().uncompressed_size())
    }
}

// TODO(ngates): why do we report the wrong thing?
impl IsSortedKernel for ListEncoding {
    fn is_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }

    fn is_strict_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }
}

register_kernel!(IsSortedKernelAdapter(ListEncoding).lift());

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
