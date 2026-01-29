// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_dtype::DType;
use vortex_error::VortexExpect;

use super::MaskedArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::arbitrary::ArbitraryArray;
use crate::arrays::arbitrary::ArbitraryConstrained;
use crate::arrays::arbitrary::ArrayConstraints;
use crate::arrays::arbitrary::ConstraintKind;
use crate::validity::Validity;

/// MaskedArray cannot generate non-nullable arrays (it always adds nullability).
pub const MASKED_CAN_GENERATE: &[ConstraintKind] = &[];

/// A wrapper type to implement `Arbitrary` for `MaskedArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryMaskedArray(pub MaskedArray);

impl<'a> Arbitrary<'a> for ArbitraryMaskedArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype: DType = u.arbitrary()?;
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryMaskedArray {
    /// Generate an arbitrary MaskedArray with the given dtype.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);

        // Generate a non-nullable child array (MaskedArray requires all-valid child)
        let child_dtype = dtype.as_nonnullable();
        let child = ArbitraryArray::arbitrary_with(u, Some(len), &child_dtype)?.0;

        // Generate random validity (must be nullable)
        let validity_bools: Vec<bool> = (0..len).map(|_| u.arbitrary()).collect::<Result<_>>()?;
        let validity = Validity::from_iter(validity_bools);

        Ok(ArbitraryMaskedArray(
            MaskedArray::try_new(child, validity)
                .vortex_expect("MaskedArray creation should succeed"),
        ))
    }

    /// Generate an arbitrary MaskedArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        _constraints: &ArrayConstraints,
    ) -> Result<Self> {
        // MaskedArray always makes the result nullable, so it can't satisfy non_nullable constraint
        Self::with_dtype(u, dtype, len)
    }
}

impl ArbitraryConstrained for MaskedArray {
    fn can_generate() -> &'static [ConstraintKind] {
        MASKED_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryMaskedArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
