// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::Canonical;
use crate::arrays::ConstantVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::compute::mask;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<MaskedVTable> for MaskedVTable {
    fn canonicalize(array: &MaskedArray) -> VortexResult<Canonical> {
        if array.child.is::<ConstantVTable>() {
            // To allow constant array to produce masked array from mask call, we have to unwrap constant here and canonicalize it first
            mask(
                array.child.to_canonical()?.as_ref(),
                &!array.validity.to_mask(array.len())?,
            )?
            .to_canonical()
        } else {
            array.masked_child()?.to_canonical()
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::Nullability;

    use super::*;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid
        ).unwrap(),
        Nullability::Nullable
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::from_iter([true, false, true])
        ).unwrap(),
        Nullability::Nullable
    )]
    fn test_canonical_nullability(
        #[case] array: MaskedArray,
        #[case] expected_nullability: Nullability,
    ) {
        let canonical = array.to_canonical();
        assert_eq!(
            canonical.as_ref().dtype().nullability(),
            expected_nullability
        );
        assert_eq!(canonical.as_ref().dtype(), array.dtype());
    }

    #[test]
    fn test_canonical_with_nulls() {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, false, true, false, true]),
        )
        .unwrap();

        let canonical = array.to_canonical();
        let prim = canonical.as_ref().to_primitive();

        // Check that null positions match validity.
        assert_eq!(prim.valid_count(), 3);
        assert!(prim.is_valid(0));
        assert!(!prim.is_valid(1));
        assert!(prim.is_valid(2));
        assert!(!prim.is_valid(3));
        assert!(prim.is_valid(4));
    }

    #[test]
    fn test_canonical_all_valid() {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid,
        )
        .unwrap();

        let canonical = array.to_canonical();
        assert_eq!(canonical.as_ref().valid_count(), 3);
        assert_eq!(
            canonical.as_ref().dtype().nullability(),
            Nullability::Nullable
        );
    }
}
