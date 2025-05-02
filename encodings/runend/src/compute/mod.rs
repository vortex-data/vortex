mod binary_numeric;
mod compare;
mod fill_null;
pub(crate) mod filter;
mod invert;
mod is_sorted;
mod min_max;
mod scalar_at;
pub(crate) mod take;
mod take_from;

use vortex_array::Array;
use vortex_array::compute::{FillNullFn, MinMaxFn, ScalarAtFn, TakeFn, TakeFromFn};
use vortex_array::vtable::ComputeVTable;

use crate::RunEndEncoding;

impl ComputeVTable for RunEndEncoding {
    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn take_from_fn(&self) -> Option<&dyn TakeFromFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::Array;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_numeric;

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
        test_numeric::<i32>(array)
    }
}
