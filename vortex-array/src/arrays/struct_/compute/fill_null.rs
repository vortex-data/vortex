// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

/// Fills null struct rows with the fields of a struct scalar.
///
/// Each field is zipped over the struct validity, so the fields of valid rows stay as they are and
/// null rows take the fields of the fill value. The result has the nullability of the fill value.
impl FillNullKernel for Struct {
    fn fill_null(
        array: ArrayView<'_, Struct>,
        fill_value: &Scalar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullability = fill_value.dtype().nullability();
        let fill_value = fill_value.cast(&array.dtype().with_nullability(nullability))?;
        let fill_struct = fill_value.as_struct();
        let fill_fields = fill_struct
            .fields_iter()
            .ok_or_else(|| vortex_err!("fill_null requires a non-null fill value"))?;

        let len = array.len();
        let is_valid = array.validity()?.to_array(len);

        let fields = array
            .iter_unmasked_fields()
            .zip(fill_fields)
            .map(|(field, fill_field)| {
                let fill = ConstantArray::new(fill_field, len).into_array();
                is_valid.zip(field.clone(), fill)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::try_new(
                array.names().clone(),
                fields,
                len,
                Validity::from(nullability),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Struct;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn names() -> FieldNames {
        FieldNames::from_iter(["id", "inner"])
    }

    fn inner_dtype() -> DType {
        DType::struct_(
            [("name", DType::Utf8(Nullability::Nullable))],
            Nullability::Nullable,
        )
    }

    /// Null rows take every field of the fill value, including a nested struct, while the
    /// fields of valid rows, and the nulls inside them, stay as they are.
    #[rstest]
    #[case::non_nullable_fill(Nullability::NonNullable)]
    #[case::nullable_fill(Nullability::Nullable)]
    fn fill_null_struct(#[case] nullability: Nullability) -> VortexResult<()> {
        let inner = |values: &[Option<&str>], validity: Validity| -> VortexResult<ArrayRef> {
            Ok(StructArray::try_new(
                FieldNames::from_iter(["name"]),
                vec![VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()],
                values.len(),
                validity,
            )?
            .into_array())
        };

        // Rows 1 and 3 are null. Row 2 is valid with a null nested struct.
        let input = StructArray::try_new(
            names(),
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4)]).into_array(),
                inner(
                    &[Some("a"), Some("b"), None, Some("d")],
                    Validity::from_iter([true, true, false, true]),
                )?,
            ],
            4,
            Validity::from_iter([true, false, true, false]),
        )?
        .into_array();

        let fill_dtype = DType::Struct(
            StructFields::new(
                names(),
                vec![
                    DType::Primitive(PType::I32, Nullability::Nullable),
                    inner_dtype(),
                ],
            ),
            nullability,
        );
        let fill = Scalar::struct_(
            fill_dtype,
            [
                Scalar::primitive(9i32, Nullability::Nullable),
                Scalar::struct_(inner_dtype(), [Scalar::utf8("fill", Nullability::Nullable)]),
            ],
        );

        let mut ctx = SESSION.create_execution_ctx();
        let result = input.fill_null(fill)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Struct>());

        let expected = StructArray::try_new(
            names(),
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), Some(9), None, Some(9)]).into_array(),
                inner(
                    &[Some("a"), Some("fill"), None, Some("fill")],
                    Validity::from_iter([true, true, false, true]),
                )?,
            ],
            4,
            Validity::from(nullability),
        )?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
