// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, VarBinView>,
    new_dtype: DType,
    new_validity: Validity,
) -> ArrayRef {
    // SAFETY: casting just changes the DType, does not affect invariants on views/buffers.
    unsafe {
        VarBinViewArray::new_handle_unchecked(
            array.views_handle().clone(),
            Arc::clone(array.data_buffers()),
            new_dtype,
            new_validity,
        )
        .into_array()
    }
}

impl CastReduce for VarBinView {
    fn cast(array: ArrayView<'_, VarBinView>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
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
        Ok(Some(build_with_validity(array, new_dtype, new_validity)))
    }
}

impl CastKernel for VarBinView {
    fn cast(
        array: ArrayView<'_, VarBinView>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array
            .validity()?
            .cast_nullability(new_nullability, array.len(), ctx)?;
        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(Some(build_with_validity(array, new_dtype, new_validity)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

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
        let varbin = VarBinViewArray::from_iter(vec![Some("a"), Some("b"), Some("c")], source);

        let res = varbin.into_array().cast(target.clone());
        assert_eq!(res.unwrap().dtype(), &target);
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn try_cast_varbin_fail(#[case] source: DType) {
        // Failure surfaces during execution via the kernel.
        let non_nullable_source = source.as_nonnullable();
        let varbin = VarBinViewArray::from_iter(vec![Some("a"), Some("b"), None], source);
        let mut ctx = SESSION.create_execution_ctx();
        let result = varbin
            .into_array()
            .cast(non_nullable_source)
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        assert!(result.is_err(), "Expected error, got: {result:?}");
    }

    #[rstest]
    #[case(VarBinViewArray::from_iter(vec![Some("hello"), Some("world"), Some("test")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("hello"), None, Some("world")], DType::Utf8(Nullability::Nullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some(b"binary".as_slice()), Some(b"data".as_slice())], DType::Binary(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some(b"test".as_slice()), None], DType::Binary(Nullability::Nullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("single")], DType::Utf8(Nullability::NonNullable)))]
    #[case(VarBinViewArray::from_iter(vec![Some("very long string that exceeds the inline size to test view functionality with multiple buffers")], DType::Utf8(Nullability::NonNullable)))]
    fn test_cast_varbinview_conformance(#[case] array: VarBinViewArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
