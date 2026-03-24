// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
pub use crate::aggregate_fn::fns::min_max::MinMaxResult;

#[deprecated(note = "use `vortex::array::aggregate_fn::fns::min_max::min_max` instead")]
pub fn min_max(array: &ArrayRef) -> VortexResult<Option<MinMaxResult>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    crate::aggregate_fn::fns::min_max::min_max(array, &mut ctx)
}
