// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::fill_null;
use vortex_array::{ArrayRef, Canonical};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

/// Apply fill_null on the canonical form of the array to get a consistent baseline.
pub fn fill_null_canonical_array(
    canonical: Canonical,
    fill_value: &Scalar,
) -> VortexResult<ArrayRef> {
    // TODO(joe): replace with baseline not using canonical
    fill_null(canonical.as_ref(), fill_value)
}
