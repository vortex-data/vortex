mod binary_numeric;
mod compare;
mod fill_null;
pub(crate) mod filter;
mod invert;
mod is_sorted;
mod scalar_at;
mod slice;
pub(crate) mod take;
mod take_from;

use vortex_array::compute::{
    BinaryNumericFn, CompareFn, FillNullFn, FilterKernelAdapter, InvertFn, IsSortedFn, KernelRef,
    ScalarAtFn, SliceFn, TakeFn, TakeFromFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl};

use crate::{RunEndArray, RunEndEncoding};

impl ArrayComputeImpl for RunEndArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(RunEndEncoding).some();
}

impl ComputeVTable for RunEndEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<&dyn Array>> {
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

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn take_from_fn(&self) -> Option<&dyn TakeFromFn<&dyn Array>> {
        Some(self)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::Array;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::test_harness::test_binary_numeric;

    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
    }

    #[test]
    fn test_runend_binary_numeric() {
        let array = ree_array().into_array();
        test_binary_numeric::<i32>(array)
    }
}
