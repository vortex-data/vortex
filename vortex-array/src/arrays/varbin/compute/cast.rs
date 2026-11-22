// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, VarBin>,
    new_dtype: DType,
    new_validity: Validity,
) -> VortexResult<ArrayRef> {
    // SAFETY: the offsets and bytes are unchanged from a valid array, and the validity was cast
    // from the array's own validity, so it has the same length.
    Ok(unsafe {
        VarBinArray::new_unchecked_from_handle(
            array.offsets().clone(),
            array.bytes_handle().clone(),
            new_dtype,
            new_validity,
        )
    }
    .into_array())
}

impl CastReduce for VarBin {
    fn cast(array: ArrayView<'_, VarBin>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(new_nullability, array.len())?
        else {
            return Ok(None);
        };
        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(Some(build_with_validity(array, new_dtype, new_validity)?))
    }
}

impl CastKernel for VarBin {
    fn cast(
        array: ArrayView<'_, VarBin>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let validity = array.validity()?;

        let new_validity = match new_nullability {
            Nullability::Nullable => validity.into_nullable(),
            Nullability::NonNullable => {
                // `VarBin` does not validate the bytes at null rows, so those bytes can be
                // invalid UTF-8. `Validity::cast_nullability` uses the statistics of the validity
                // array, which can come from untrusted input. A forged statistic would then show
                // unchecked bytes as `Utf8` values. Read the validity values instead.
                let mask = validity.execute_mask(array.len(), ctx)?;

                vortex_ensure!(
                    mask.all_true(),
                    "Cannot cast array with invalid values to non-nullable type."
                );

                Validity::NonNullable
            }
        };

        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(Some(build_with_validity(array, new_dtype, new_validity)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(
        DType::Utf8(Nullability::Nullable),
        DType::Utf8(Nullability::NonNullable)
    )]
    #[case(
        DType::Binary(Nullability::Nullable),
        DType::Binary(Nullability::NonNullable)
    )]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable)
    )]
    #[case(
        DType::Binary(Nullability::NonNullable),
        DType::Binary(Nullability::Nullable)
    )]
    fn try_cast_varbin_nullable(#[case] source: DType, #[case] target: DType) {
        let varbin = VarBinArray::from_iter(vec![Some("a"), Some("b"), Some("c")], source);

        let res = varbin.into_array().cast(target.clone());
        assert_arrays_eq!(
            res.unwrap(),
            VarBinViewArray::from_iter([Some("a"), Some("b"), Some("c")], target),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn try_cast_varbin_fail(#[case] source: DType) {
        // Failure surfaces during execution via the kernel.
        let non_nullable_source = source.as_nonnullable();
        let varbin = VarBinArray::from_iter(vec![Some("a"), Some("b"), None], source);
        let mut ctx = SESSION.create_execution_ctx();
        let result = varbin
            .into_array()
            .cast(non_nullable_source)
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        assert!(result.is_err(), "Expected error, got: {result:?}");
    }

    #[test]
    fn cast_to_non_nullable_ignores_forged_min_statistic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // Only the null row holds bytes that are not valid UTF-8.
        let values = Buffer::copy_from([b'a', 0xFF, b'b']);
        let offsets = buffer![0i32, 1, 2, 3].into_array();

        let is_valid = BoolArray::from_iter([true, false, true]).into_array();
        is_valid.statistics().set(Stat::Min, Precision::exact(true));

        let varbin = VarBinArray::try_new(
            offsets,
            values,
            DType::Utf8(Nullability::Nullable),
            Validity::Array(is_valid),
            &mut ctx,
        )?;

        let result = varbin
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx));

        assert!(result.is_err(), "Expected error, got: {result:?}");
        Ok(())
    }

    #[rstest]
    #[case(VarBinArray::from_iter(vec![Some("hello"), Some("world"), Some("test")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinArray::from_iter(vec![Some("hello"), None, Some("world")], DType::Utf8(Nullability::Nullable)))]
    #[case(VarBinArray::from_iter(vec![Some(b"binary".as_slice()), Some(b"data".as_slice())], DType::Binary(Nullability::NonNullable)))]
    #[case(VarBinArray::from_iter(vec![Some(b"test".as_slice()), None], DType::Binary(Nullability::Nullable)))]
    #[case(VarBinArray::from_iter(vec![Some("single")], DType::Utf8(Nullability::NonNullable)))]
    fn test_cast_varbin_conformance(#[case] array: VarBinArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
