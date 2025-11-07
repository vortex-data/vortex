// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::bool::BoolVectorMut;
use crate::{Scalar, ScalarOps, VectorMut, VectorMutOps};

/// A scalar value for boolean types.
pub struct BoolScalar(Option<bool>);

impl From<Option<bool>> for BoolScalar {
    fn from(value: Option<bool>) -> Self {
        Self(value)
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
