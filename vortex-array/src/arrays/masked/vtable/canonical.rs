// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
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
    ) -> VortexResult<()> {
        let canonical = array.to_canonical()?;
        assert_eq!(
            canonical.as_ref().dtype().nullability(),
            expected_nullability
        );
        assert_eq!(canonical.as_ref().dtype(), array.dtype());
        Ok(())
    }

    #[test]
    fn test_canonical_with_nulls() -> VortexResult<()> {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, false, true, false, true]),
        )
        .unwrap();

        let canonical = array.to_canonical()?;
        let prim = canonical.as_ref().to_primitive();

        // Check that null positions match validity.
        assert_eq!(prim.valid_count().unwrap(), 3);
        assert!(prim.is_valid(0).unwrap());
        assert!(!prim.is_valid(1).unwrap());
        assert!(prim.is_valid(2).unwrap());
        assert!(!prim.is_valid(3).unwrap());
        assert!(prim.is_valid(4).unwrap());
        Ok(())
    }

    #[test]
    fn test_canonical_all_valid() -> VortexResult<()> {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid,
        )
        .unwrap();

        let canonical = array.to_canonical()?;
        assert_eq!(canonical.as_ref().valid_count().unwrap(), 3);
        assert_eq!(
            canonical.as_ref().dtype().nullability(),
            Nullability::Nullable
        );
        Ok(())
    }
}
