// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::struct_::StructVector;
use crate::{ScalarOps, VectorMut, VectorOps};

/// Represents a struct scalar value.
///
/// The inner value is a StructVector with length 1.
pub struct StructScalar(StructVector);

impl ScalarOps for StructScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn repeat(&self, n: usize) -> VectorMut {
        todo!()
    }
}
