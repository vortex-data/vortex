// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::mask as mask_fn;
use vortex_array::{ArrayRef, Canonical};
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// Apply mask on the canonical form of the array to get a consistent baseline.
pub fn mask_canonical_array(canonical: Canonical, mask: &Mask) -> VortexResult<ArrayRef> {
    // TODO(joe): replace with baseline not using canonical
    mask_fn(canonical.as_ref(), mask)
}
