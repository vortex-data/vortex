// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{MaskedArray, MaskedVTable};
use crate::compute::{TakeKernel, TakeKernelAdapter, fill_null, take};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl TakeKernel for MaskedVTable {
    fn take(&self, array: &MaskedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_child = if !indices.all_valid() {
            // This is safe because we'll mask out these positions in the validity
            let filled_take = fill_null(
                indices,
                &Scalar::default_value(indices.dtype().clone().as_nonnullable()),
            )?;
            take(&array.child, &filled_take)?
        } else {
            take(&array.child, indices)?
        };

        // Compute the new validity by taking from array's validity and merging with indices validity
        let taken_validity = array.validity().take(indices)?;

        // Construct new MaskedArray
        Ok(MaskedArray::try_new(taken_child, taken_validity)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(MaskedVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::{MaskedArray, PrimitiveArray};
    use crate::compute::conformance::take::test_take_conformance;
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
    fn test_take_masked_conformance(#[case] array: MaskedArray) {
        test_take_conformance(array.as_ref());
    }
}
