mod filter;
mod mask;

use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{
    IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts, IsSortedKernel,
    IsSortedKernelAdapter, MinMaxKernel, MinMaxKernelAdapter, MinMaxResult,
};
use crate::register_kernel;

impl IsConstantKernel for ListVTable {
    fn is_constant(
        &self,
        _array: &ListArray,
        _opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        // TODO(adam): Do we want to fallback to arrow here?
        Ok(None)
    }
}

register_kernel!(IsConstantKernelAdapter(ListVTable).lift());

impl MinMaxKernel for ListVTable {
    fn min_max(&self, _array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // TODO(joe): Implement list min max
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(ListVTable).lift());

// TODO(ngates): why do we report the wrong thing?
impl IsSortedKernel for ListVTable {
    fn is_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }

    fn is_strict_sorted(&self, _array: &ListArray) -> VortexResult<bool> {
        Ok(false)
    }
}

register_kernel!(IsSortedKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod test {
    use crate::IntoArray;
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

        test_mask(array.as_ref());
    }
}
