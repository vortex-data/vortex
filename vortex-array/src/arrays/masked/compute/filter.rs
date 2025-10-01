// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{MaskedArray, MaskedVTable};
use crate::compute::{FilterKernel, FilterKernelAdapter, filter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl FilterKernel for MaskedVTable {
    fn filter(&self, array: &MaskedArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // Filter the validity to get the new validity
        let filtered_validity = array.validity().filter(mask)?;

        // Filter the child array
        // The child is guaranteed to have no nulls, so filtering it is straightforward
        let filtered_child = filter(&array.child, mask)?;

        // Construct new MaskedArray
        Ok(MaskedArray::try_new(filtered_child, filtered_validity)?.into_array())
    }
}

register_kernel!(FilterKernelAdapter(MaskedVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::{MaskedArray, PrimitiveArray};
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_filter_masked_conformance(#[case] array: MaskedArray) {
        test_filter_conformance(array.as_ref());
    }
}
