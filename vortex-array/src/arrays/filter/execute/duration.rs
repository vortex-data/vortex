// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use crate::arrays::DurationArray;
use crate::arrays::duration::DurationArrayExt;
use crate::arrays::duration::DurationArraySlotsExt;

pub(super) fn filter_duration(array: &DurationArray, mask: &MaskValuesRef) -> DurationArray {
    let values = array
        .values()
        .filter(Mask::Values(MaskValuesRef::clone(mask)))
        .vortex_expect("duration values must support filtering");
    DurationArray::try_new(array.unit(), values)
        .vortex_expect("filtering preserves the duration value dtype")
}
