// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::null::NullVectorMut;
use crate::{Scalar, ScalarOps, VectorMut};

/// Represents a null scalar value.
#[derive(Debug)]
pub struct NullScalar;

impl ScalarOps for NullScalar {
    fn is_valid(&self) -> bool {
        false
    }

    fn repeat(&self, n: usize) -> VectorMut {
        NullVectorMut::new(n).into()
    }
}

impl From<NullScalar> for Scalar {
    fn from(val: NullScalar) -> Self {
        Scalar::Null(val)
    }
}
