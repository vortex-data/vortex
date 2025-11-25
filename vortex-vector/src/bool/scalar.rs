// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorMutOps;
use crate::bool::BoolVectorMut;

/// A scalar value for boolean types.
#[derive(Debug)]
pub struct BoolScalar(Option<bool>);

impl BoolScalar {
    /// Creates a new bool scalar with the given value.
    pub fn new(value: Option<bool>) -> Self {
        Self(value)
    }

    /// Returns the value of the bool scalar, or `None` if the scalar is null.
    pub fn value(&self) -> Option<bool> {
        self.0
    }
}

impl ScalarOps for BoolScalar {
    fn is_valid(&self) -> bool {
        self.0.is_some()
    }

    fn repeat(&self, n: usize) -> VectorMut {
        let mut vec = BoolVectorMut::with_capacity(n);
        match self.0 {
            None => vec.append_nulls(n),
            Some(value) => vec.append_values(value, n),
        }
        vec.into()
    }
}

impl From<BoolScalar> for Scalar {
    fn from(val: BoolScalar) -> Self {
        Scalar::Bool(val)
    }
}
