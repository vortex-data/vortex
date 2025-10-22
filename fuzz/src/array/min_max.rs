// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::compute::{MinMaxResult, min_max};
use vortex_error::VortexResult;

/// Compute min_max on the canonical form of the array to get a consistent baseline.
pub fn min_max_canonical_array(canonical: Canonical) -> VortexResult<Option<MinMaxResult>> {
    // TODO(joe): replace with baseline not using canonical
    min_max(canonical.as_ref())
}
