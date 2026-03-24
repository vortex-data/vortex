// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

/// Computes whether an array is sorted in non-decreasing order.
///
/// **Deprecated**: Use [`crate::aggregate_fn::fns::is_sorted::is_sorted`] instead.
#[deprecated(note = "Use crate::aggregate_fn::fns::is_sorted::is_sorted instead")]
pub fn is_sorted(array: &ArrayRef) -> VortexResult<Option<bool>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    Ok(Some(crate::aggregate_fn::fns::is_sorted::is_sorted(
        array, &mut ctx,
    )?))
}

/// Computes whether an array is strictly sorted in increasing order.
///
/// **Deprecated**: Use [`crate::aggregate_fn::fns::is_sorted::is_strict_sorted`] instead.
#[deprecated(note = "Use crate::aggregate_fn::fns::is_sorted::is_strict_sorted instead")]
pub fn is_strict_sorted(array: &ArrayRef) -> VortexResult<Option<bool>> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    Ok(Some(crate::aggregate_fn::fns::is_sorted::is_strict_sorted(
        array, &mut ctx,
    )?))
}
