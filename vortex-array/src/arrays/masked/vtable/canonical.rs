// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    fn masked_all_valid() -> MaskedArray {
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid,
        )
        .expect("valid masked array")
    }

    fn masked_with_nulls() -> MaskedArray {
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::from_iter([true, false, true]),
        )
        .expect("valid masked array")
    }

    #[rstest]
    #[case(masked_all_valid(), Nullability::Nullable)]
    #[case(masked_with_nulls(), Nullability::Nullable)]
    fn test_canonical_nullability(
        #[case] array: MaskedArray,
        #[case] expected_nullability: Nullability,
    ) -> VortexResult<()> {
        let canonical = array
            .clone()
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?;
        assert_eq!(canonical.dtype().nullability(), expected_nullability);
        assert_eq!(canonical.dtype(), array.dtype());
        Ok(())
    }

    #[test]
    fn test_canonical_with_nulls() -> VortexResult<()> {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, false, true, false, true]),
        )?;

        let canonical = array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?;
        let prim = canonical.into_primitive();

        // Check that null positions match validity.
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(prim.valid_count(&mut ctx)?, 3);
        assert!(prim.is_valid(0, &mut array_session().create_execution_ctx())?);
        assert!(!prim.is_valid(1, &mut array_session().create_execution_ctx())?);
        assert!(prim.is_valid(2, &mut array_session().create_execution_ctx())?);
        assert!(!prim.is_valid(3, &mut array_session().create_execution_ctx())?);
        assert!(prim.is_valid(4, &mut array_session().create_execution_ctx())?);
        Ok(())
    }

    #[test]
    fn test_canonical_all_valid() -> VortexResult<()> {
        let array = MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid,
        )?;

        let canonical = array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?;
        assert_eq!(canonical.dtype().nullability(), Nullability::Nullable);
        assert_eq!(
            canonical
                .into_array()
                .valid_count(&mut array_session().create_execution_ctx())?,
            3
        );
        Ok(())
    }
}
