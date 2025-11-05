// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{private, VectorMut};

pub trait ScalarOps: private::Sealed + Sized {
    /// Returns true if the scalar is valid (not null).
    fn is_valid(&self) -> bool;

    /// Returns true if the scalar is null.
    fn is_invalid(&self) -> bool {
        !self.is_valid()
    }

    /// Creates a new vector with n repetitions of this scalar.
    fn repeat(&self, n: usize) -> VectorMut;
}
