// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unit-vector extension type for fixed-length float vectors.
//!
//! A [`UnitVector`] uses the same fixed-size-list storage as
//! [`Vector`](crate::vector::Vector), but every non-null row is either exactly zero or has an L2
//! norm within [`unit_norm_tolerance`](crate::unit_norm_tolerance) of one. Use
//! [`try_new_unit_vector_array`](UnitVector::try_new_unit_vector_array) at untrusted construction
//! boundaries.

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexResult;

use crate::encodings::normalized::validate_normalized_rows;

mod arrow;
pub use arrow::ARROW_UNIT_VECTOR_EXTENSION_NAME;

mod matcher;
pub use matcher::AnyUnitVector;

mod vtable;

/// A fixed-length float vector that is unit norm within the configured tolerance, or exactly zero.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct UnitVector;

impl UnitVector {
    /// Constructs a [`UnitVector`] array after validating every non-null row.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtype is incompatible or any non-null row is neither
    /// exactly zero nor unit norm within the configured tolerance.
    pub fn try_new_unit_vector_array(
        storage: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // SAFETY: The array is validated immediately below before it is returned to the caller.
        let array = unsafe { Self::new_unchecked(storage)? };
        validate_normalized_rows(&array, None, ctx)?;

        Ok(array)
    }

    /// Constructs a [`UnitVector`] array without validating its row values.
    ///
    /// # Safety
    ///
    /// Every non-null row must be exactly zero or have an L2 norm within
    /// [`unit_norm_tolerance`](crate::unit_norm_tolerance) of one. Violating this contract can
    /// produce incorrect results in operations that use the refinement for approximate compute
    /// shortcuts; it does not cause memory unsafety.
    pub unsafe fn new_unchecked(storage: ArrayRef) -> VortexResult<ArrayRef> {
        ExtensionArray::try_new_from_vtable(UnitVector, EmptyMetadata, storage)
            .map(|array| array.into_array())
    }
}

#[cfg(test)]
mod tests;
