// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
mod like;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::Array;
    use vortex_array::arrays::{ConstantArray, VarBinViewArray};
    use vortex_array::compute::{CompareKernel, Operator, cast};
    use vortex_dtype::{DType, Nullability};
    use vortex_error::VortexResult;
    use vortex_scalar::Scalar;

    use crate::{FSSTViewArray, FSSTViewEncoding, FSSTViewVTable};

    #[rstest::fixture]
    fn strings() -> FSSTViewArray {
        let canonical =
            VarBinViewArray::from_iter_str(["a", "b", "super duper long string abcdefg"])
                .to_canonical();

        FSSTViewEncoding
            .encode(&canonical, None)
            .unwrap()
            .unwrap()
            .as_::<FSSTViewVTable>()
            .clone()
    }

    #[rstest::fixture]
    fn nullable_bin() -> FSSTViewArray {
        let canonical = VarBinViewArray::from_iter_nullable_bin([
            None,
            Some("b"),
            None,
            Some("super duper long string abcdefg"),
        ])
        .to_canonical();

        FSSTViewEncoding
            .encode(&canonical, None)
            .unwrap()
            .unwrap()
            .as_::<FSSTViewVTable>()
            .clone()
    }

    #[rstest]
    fn test_compare(strings: FSSTViewArray) -> VortexResult<()> {
        // Simple short string
        let const_a =
            ConstantArray::new(Scalar::utf8("a", Nullability::NonNullable), strings.len());

        let eq_a = FSSTViewVTable
            .compare(&strings, const_a.as_ref(), Operator::Eq)?
            .unwrap();

        assert_eq!(
            [eq_a.scalar_at(0), eq_a.scalar_at(1), eq_a.scalar_at(2)],
            [true.into(), false.into(), false.into()]
        );

        // Not Eq
        let not_eq_a = FSSTViewVTable
            .compare(&strings, const_a.as_ref(), Operator::NotEq)?
            .unwrap();

        assert_eq!(
            [
                not_eq_a.scalar_at(0),
                not_eq_a.scalar_at(1),
                not_eq_a.scalar_at(2)
            ],
            [false.into(), true.into(), true.into()]
        );

        // Outlined string - eq
        let const_long = ConstantArray::new(
            Scalar::utf8("super duper long string abcdefg", Nullability::NonNullable),
            strings.len(),
        );

        let eq_long = FSSTViewVTable
            .compare(&strings, const_long.as_ref(), Operator::Eq)?
            .unwrap();

        assert_eq!(
            [
                eq_long.scalar_at(0),
                eq_long.scalar_at(1),
                eq_long.scalar_at(2)
            ],
            [false.into(), false.into(), true.into()]
        );

        // Outlined string - not eq
        let not_eq_long = FSSTViewVTable
            .compare(&strings, const_long.as_ref(), Operator::NotEq)?
            .unwrap();

        assert_eq!(
            [
                not_eq_long.scalar_at(0),
                not_eq_long.scalar_at(1),
                not_eq_long.scalar_at(2)
            ],
            [true.into(), true.into(), false.into()]
        );

        // Pushdown other operators not supported yet
        for unsupported in [Operator::Lt, Operator::Lte, Operator::Gt, Operator::Gte] {
            assert!(
                FSSTViewVTable
                    .compare(&strings, const_long.as_ref(), unsupported)?
                    .is_none()
            );
        }

        Ok(())
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_cast_succeed(strings: FSSTViewArray, #[case] target: DType) -> VortexResult<()> {
        let result = cast(strings.as_ref(), &target)?;
        assert_eq!(result.dtype(), &target);

        Ok(())
    }

    #[rstest]
    #[case(DType::Binary(Nullability::NonNullable))]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Utf8(Nullability::NonNullable))]
    fn test_cast_fail(nullable_bin: FSSTViewArray, #[case] target: DType) {
        assert!(cast(nullable_bin.as_ref(), &target).is_err());
    }
}
