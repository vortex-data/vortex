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

impl FillNullKernel for Extension {
    fn fill_null(
        array: ArrayView<'_, Extension>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let storage_fill = fill_value.as_extension().to_storage_scalar();
        let filled_storage = array
            .storage_array()
            .clone()
            .fill_null(storage_fill)?
            .execute::<ArrayRef>(ctx)?;
        let ext_dtype = array
            .ext_dtype()
            .with_nullability(filled_storage.dtype().nullability());
        Ok(Some(
            ExtensionArray::new(ext_dtype, filled_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;

    #[test]
    fn fill_null_extension_array() -> VortexResult<()> {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let storage =
            PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), None]).into_array();
        let array = ExtensionArray::new(ext_dtype.clone(), storage).into_array();

        let fill_value = Scalar::extension_ref(
            ext_dtype.with_nullability(Nullability::NonNullable),
            Scalar::from(42i64),
        );
        let filled = array.fill_null(fill_value)?;

        assert!(matches!(filled.dtype(), DType::Extension(e) if !e.storage_dtype().is_nullable()));
        let expected = ExtensionArray::new(
            ext_dtype.with_nullability(Nullability::NonNullable),
            PrimitiveArray::from_iter([1i64, 42, 3, 42]).into_array(),
        );
        assert_arrays_eq!(filled, expected);
        Ok(())
    }
}
