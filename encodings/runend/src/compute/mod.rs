mod binary_numeric;
mod compare;
mod fill_null;
pub(crate) mod filter;
mod invert;
mod scalar_at;
mod slice;
pub(crate) mod take;

use vortex_array::compute::{
    BinaryNumericFn, CompareFn, FillNullFn, FilterFn, InvertFn, ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::ArrayData;

use crate::RunEndEncoding;

impl ComputeVTable for RunEndEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<ArrayData>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<ArrayData>> {
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

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::test_harness::test_binary_numeric;
    use vortex_array::IntoArrayData;

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
