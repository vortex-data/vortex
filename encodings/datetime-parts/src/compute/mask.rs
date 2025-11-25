// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::compute::MaskKernel;
use vortex_array::compute::MaskKernelAdapter;
use vortex_array::compute::mask;
use vortex_array::register_kernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

impl MaskKernel for DateTimePartsVTable {
    fn mask(&self, array: &DateTimePartsArray, mask_array: &Mask) -> VortexResult<ArrayRef> {
        // DateTimePartsArray has specific constraints:
        // - days nullability must match the dtype
        // - seconds and subseconds must always be non-nullable
        //
        // When masking, we can't make seconds/subseconds nullable.
        // Instead, we'll keep the same values but the overall array becomes nullable
        // through the days component.

        let masked_days = mask(array.days(), mask_array)?;

        // Keep seconds and subseconds unchanged since they must remain non-nullable
        let seconds = array.seconds().clone();
        let subseconds = array.subseconds().clone();

        // Update the dtype to reflect the new nullability of days
        let new_dtype = if masked_days.dtype().is_nullable() {
            array.dtype().as_nullable()
        } else {
            array.dtype().clone()
        };

        DateTimePartsArray::try_new(new_dtype, masked_days, seconds, subseconds)
            .map(|a| a.to_array())
    }
}

register_kernel!(MaskKernelAdapter(DateTimePartsVTable).lift());
