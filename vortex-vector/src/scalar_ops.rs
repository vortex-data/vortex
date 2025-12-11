// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::VectorMut;
use crate::private;

/// Trait for scalar operations.
pub trait ScalarOps: private::Sealed + Sized + Into<Scalar> {
    /// Returns true if the scalar is valid (not null).
    fn is_valid(&self) -> bool;

    /// Returns true if the scalar is null.
    fn is_null(&self) -> bool {
        !self.is_valid()
    }

    /// Intersect the validity of this scalar with the provided mask value.
    ///
    /// If the mask is true, the scalar's validity remains unchanged.
    /// If the mask is false, the resulting scalar is null.
    fn mask_validity(&mut self, mask: bool);

    /// Creates a new vector with n repetitions of this scalar.
    fn repeat(&self, n: usize) -> VectorMut;
}
