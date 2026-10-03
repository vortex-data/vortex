// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, VarBinView>,
    new_dtype: DType,
    new_validity: Validity,
) -> ArrayRef {
    // SAFETY: views and buffers are unchanged. Null views may have invalid bounds or UTF-8, so
    // removing nullability requires every source row to be valid. The caller proves this from
    // the validity representation or its executed bits before exposing those views.
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
        let validity = array.validity()?;
        let new_validity = match new_nullability {
            Nullability::NonNullable => {
                if !validity.execute_no_nulls(array.len(), ctx)? {
                    vortex_bail!(InvalidArgument: "Cannot cast array with invalid values to non-nullable type.");
                }
                Validity::NonNullable
            }
            Nullability::Nullable => validity.into_nullable(),
        };
        let new_dtype = array.dtype().with_nullability(new_nullability);
        Ok(Some(build_with_validity(array, new_dtype, new_validity)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::min::MIN_SKIP_NANS;
    use crate::arrays::BoolArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[rstest]
    #[case::ascii_utf8(DType::Utf8(Nullability::Nullable), b'x')]
    #[case::ascii_binary(DType::Binary(Nullability::Nullable), b'x')]
    #[case::invalid_utf8(DType::Utf8(Nullability::Nullable), 0xff)]
    #[case::non_utf8_null_binary(DType::Binary(Nullability::Nullable), 0xff)]
    fn non_nullable_cast_checks_validity_bits(
        #[case] dtype: DType,
        #[case] null_byte: u8,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = BoolArray::from_iter([false, true]).into_array();
        let array = VarBinArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![null_byte, b'a'],
            dtype.clone(),
            Validity::Array(validity.clone()),
        )?
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)?;
        let donor = BoolArray::from_iter([true, true]).into_array();
        donor
            .aggregations()
            .compute_result(&MIN_SKIP_NANS, &mut ctx)?;
        // The donor minimum is correct for a different input, so it cannot prove this validity.
        validity.aggregations().inherit_from(donor.aggregations());

        let casted = array.into_array().cast(dtype.as_nonnullable())?;
        assert!(casted.execute::<Canonical>(&mut ctx).is_err());
        Ok(())
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn non_nullable_cast_accepts_all_valid_bits(#[case] dtype: DType) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = BoolArray::from_iter([true, true]).into_array();
        let array = VarBinArray::try_new(
            buffer![0u32, 1, 2].into_array(),
            buffer![b'x', b'a'],
            dtype.clone(),
            Validity::Array(validity.clone()),
        )?
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)?;
        let donor = BoolArray::from_iter([false, true]).into_array();
        donor
            .aggregations()
            .compute_result(&MIN_SKIP_NANS, &mut ctx)?;
        // The donor minimum is correct for a different input, so it cannot prove this validity.
        validity.aggregations().inherit_from(donor.aggregations());

        let target = dtype.as_nonnullable();
        let casted = array.into_array().cast(target.clone())?;
        let executed = casted.execute::<Canonical>(&mut ctx)?;
        let expected = VarBinViewArray::from_iter([Some("x"), Some("a")], target);
        assert_arrays_eq!(executed.into_array(), expected, &mut ctx);
        Ok(())
    }

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
