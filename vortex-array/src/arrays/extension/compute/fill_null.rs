// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;

/// Fills the nulls of the storage array with the storage value of the fill scalar, then wraps the
/// result in the extension type.
impl FillNullKernel for Extension {
    fn fill_null(
        array: ArrayView<'_, Extension>,
        fill_value: &Scalar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullability = fill_value.dtype().nullability();
        let fill_value = fill_value.cast(&array.dtype().with_nullability(nullability))?;
        let storage_fill = fill_value.as_extension().to_storage_scalar();

        let storage = array.storage_array().fill_null(storage_fill)?;
        let ext_dtype = array
            .ext_dtype()
            .with_nullability(storage.dtype().nullability());

        Ok(Some(
            ExtensionArray::try_new(ext_dtype, storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn timestamp(nullability: Nullability) -> ExtDTypeRef {
        Timestamp::new(TimeUnit::Milliseconds, nullability).erased()
    }

    /// Null timestamps take the fill timestamp, and the result has the fill's nullability.
    #[rstest]
    #[case::non_nullable_fill(Nullability::NonNullable)]
    #[case::nullable_fill(Nullability::Nullable)]
    fn fill_null_timestamps(#[case] nullability: Nullability) -> VortexResult<()> {
        let input = ExtensionArray::new(
            timestamp(Nullability::Nullable),
            PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), None]).into_array(),
        )
        .into_array();
        let fill = Scalar::extension_ref(
            timestamp(nullability),
            Scalar::primitive(99i64, nullability),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let result = input.fill_null(fill)?;

        let expected = ExtensionArray::new(
            timestamp(nullability),
            PrimitiveArray::new(buffer![10i64, 99, 30, 99], Validity::from(nullability))
                .into_array(),
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
