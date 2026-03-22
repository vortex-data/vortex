// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

#[deprecated(note = "use `vortex::array::aggregate_fn::fns::nan_count::nan_count` instead")]
pub fn nan_count(array: &ArrayRef) -> VortexResult<usize> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    crate::aggregate_fn::fns::nan_count::nan_count(array, &mut ctx)
}
