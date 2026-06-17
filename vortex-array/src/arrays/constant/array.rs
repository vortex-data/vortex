// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use crate::array::Array;
use crate::array::ArrayParts;
use crate::arrays::Constant;
use crate::dtype::DType;
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

    /// Construct an entirely-null constant array of the given nullable `dtype` and `len`.
    ///
    /// This is the canonical representation of an all-null array: a single null scalar repeated
    /// `len` times, carrying neither a values buffer nor a validity child.
    pub fn null(dtype: DType, len: usize) -> Self {
        Self::new(Scalar::null(dtype), len)
    }
}
