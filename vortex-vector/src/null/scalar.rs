// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::null::NullVector;
use crate::{Scalar, ScalarOps, Vector};

/// Represents a null scalar value.
#[derive(Debug)]
pub struct NullScalar;

impl ScalarOps for NullScalar {
    fn is_valid(&self) -> bool {
        false
    }

    fn repeat(&self, n: usize) -> Vector {
        NullVector::new(n).into()
    }
}

impl From<NullScalar> for Scalar {
    fn from(val: NullScalar) -> Self {
        Scalar::Null(val)
    }
}
