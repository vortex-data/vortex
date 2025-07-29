// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compare;
mod is_constant;

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, filter, take,
};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{FoRArray, FoRVTable};

impl TakeKernel for FoRVTable {
    fn take(&self, array: &FoRArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            take(array.encoded(), indices)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(TakeKernelAdapter(FoRVTable).lift());

impl FilterKernel for FoRVTable {
    fn filter(&self, array: &FoRArray, mask: &Mask) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            filter(array.encoded(), mask)?,
            array.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(FilterKernelAdapter(FoRVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_scalar::Scalar;

    use crate::FoRArray;

    #[test]
    fn test_filter_for_array() {
        // Test with i32 values
        let values = PrimitiveArray::from_iter([100i32, 101, 102, 103, 104]);
        let reference = Scalar::from(100i32);
        let for_array = FoRArray::try_new(values.into_array(), reference).unwrap();
        test_filter_conformance(for_array.as_ref());

        // Test with u64 values
        let values = PrimitiveArray::from_iter([1000u64, 1001, 1002, 1003, 1004]);
        let reference = Scalar::from(1000u64);
        let for_array = FoRArray::try_new(values.into_array(), reference).unwrap();
        test_filter_conformance(for_array.as_ref());

        // Test with nullable values
        let values =
            PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]);
        let reference = Scalar::from(50i16);
        let for_array = FoRArray::try_new(values.into_array(), reference).unwrap();
        test_filter_conformance(for_array.as_ref());
    }
}
