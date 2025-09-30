// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::arrays::{MaskedArray, MaskedVTable, PrimitiveArray};
use crate::canonical::ToCanonical;
use crate::compute::{TakeKernel, TakeKernelAdapter, take};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl TakeKernel for MaskedVTable {
    fn take(&self, array: &MaskedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Strip nulls from indices to get non-nullable indices for taking from child
        // The child is guaranteed to have no nulls, so we can only take with non-null indices
        let stripped_indices = if indices.dtype().is_nullable() {
            // Create a non-nullable version of the indices by replacing nulls with 0
            // This is safe because we'll mask out these positions in the validity
            let prim = indices.to_primitive();
            match_each_native_ptype!(prim.ptype(), |P| {
                PrimitiveArray::new(prim.into_buffer::<P>(), Validity::NonNullable).into_array()
            })
        } else {
            indices.to_owned()
        };

        // Take from the child using the stripped indices
        let taken_child = take(&array.child, &stripped_indices)?;

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
            PrimitiveArray::from_iter([100i32]).into_array(),
            Validity::NonNullable
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
