// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::scalar::Scalar;

#[deprecated(note = "use `vortex::array::aggregate_fn::fns::sum::sum` instead")]
pub fn sum(array: &ArrayRef) -> VortexResult<Scalar> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    crate::aggregate_fn::fns::sum::sum(array, &mut ctx)
}
