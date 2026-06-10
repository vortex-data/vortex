// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrow::ArrowSessionExt;
use vortex_error::VortexResult;

/// Round-trip an array through Arrow: execute it into an Arrow array and import it back.
///
/// The Arrow field is inferred from the array's [`DType`](vortex_array::dtype::DType) and used
/// for both directions, so a successful round trip must preserve the logical type, including
/// nullability. The result is logically identical to the input.
pub fn arrow_roundtrip_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    let field = arrow.to_arrow_field("item", array.dtype())?;
    let arrow_array = arrow.execute_arrow(array.clone(), Some(&field), ctx)?;
    arrow.from_arrow_array(arrow_array, &field)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::arrow_roundtrip_array;

    #[test]
    fn test_arrow_roundtrip_primitive() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();

        let result = arrow_roundtrip_array(&array, &mut ctx)?;

        assert_arrays_eq!(array, result);
        Ok(())
    }

    #[test]
    fn test_arrow_roundtrip_struct() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = StructArray::try_new(
            ["a", "b"].into(),
            vec![
                BoolArray::from_iter([true, false, true]).into_array(),
                VarBinViewArray::from_iter_nullable_str([Some("x"), None, Some("z")]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        let result = arrow_roundtrip_array(&array, &mut ctx)?;

        assert_arrays_eq!(array, result);
        Ok(())
    }
}
