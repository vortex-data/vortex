// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`TakeArray`].
//!
//! The main entrypoint is [`execute_take`] which takes elements from any [`Canonical`] array.

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::TakeArray;
use crate::compute::take;

/// Check for some fast-path execution conditions before calling [`execute_take`].
pub(super) fn execute_take_fast_paths(
    array: &TakeArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
    // If the indices are empty, the output is empty.
    if array.indices.is_empty() {
        return Ok(Some(Canonical::empty(array.dtype())));
    }

    // If all indices are invalid (null), return an array of nulls
    if array.indices.all_invalid()? {
        return Ok(Some(
            ConstantArray::new(
                Scalar::null(array.child.dtype().as_nullable()),
                array.indices.len(),
            )
            .into_array()
            .execute(ctx)?,
        ));
    }

    // Also check if the source array itself is completely null
    if array.child.validity_mask()?.true_count() == 0 {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), array.indices.len())
                .into_array()
                .execute(ctx)?,
        ));
    }

    Ok(None)
}

/// Take elements from a canonical array at the given indices, returning a new canonical array.
pub(super) fn execute_take(canonical: Canonical, indices: ArrayRef) -> VortexResult<Canonical> {
    // For now, delegate to the compute take function and canonicalize the result
    let taken = take(canonical.as_ref(), indices.as_ref())?;
    taken.to_canonical()
}
