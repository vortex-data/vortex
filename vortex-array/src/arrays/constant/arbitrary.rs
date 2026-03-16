// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;

use super::ConstantArray;
use crate::dtype::DType;
use crate::scalar::arbitrary::random_scalar;

/// A wrapper type to implement `Arbitrary` for `ConstantArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryConstantArray(pub ConstantArray);

impl<'a> Arbitrary<'a> for ArbitraryConstantArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype: DType = u.arbitrary()?;
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryConstantArray {
    /// Generate an arbitrary ConstantArray with the given dtype.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        let scalar = random_scalar(u, dtype)?;
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        Ok(ArbitraryConstantArray(ConstantArray::new(scalar, len)))
    }
}
