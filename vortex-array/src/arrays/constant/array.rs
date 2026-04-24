// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use crate::array::Array;
use crate::array::ArrayParts;
use crate::arrays::Constant;
use crate::scalar::Scalar;

#[derive(Clone, Debug)]
pub struct ConstantData {
    pub(super) scalar: Scalar,
}

impl Display for ConstantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scalar: {}", self.scalar)
    }
}

impl ConstantData {
    pub fn new<S>(scalar: S) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        Self { scalar }
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn into_parts(self) -> Scalar {
        self.scalar
    }
}

impl Array<Constant> {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let dtype = scalar.dtype().clone();
        let data = ConstantData::new(scalar);
        unsafe { Array::from_parts_unchecked(ArrayParts::new(Constant, dtype, len, data)) }
    }
}
