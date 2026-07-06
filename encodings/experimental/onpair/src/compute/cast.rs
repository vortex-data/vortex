// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

/// Adjust nullability without touching any encoded payload — we only rewrap
/// into a new outer DType.
fn build_with_validity(
    array: ArrayView<'_, OnPair>,
    dtype: &DType,
    validity: Validity,
) -> ArrayRef {
    unsafe {
        OnPair::new_unchecked(
            dtype.clone(),
            array.dict_bytes_handle().clone(),
            array.dict_offsets().clone(),
            array.codes().clone(),
            array.codes_offsets().clone(),
            array.uncompressed_lengths().clone(),
            validity,
        )
    }
    .into_array()
}

impl CastReduce for OnPair {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }
        let validity = array.array().validity()?;
        let Some(new_validity) =
            validity.trivially_cast_nullability(dtype.nullability(), array.array().len())?
        else {
            return Ok(None);
        };
        Ok(Some(build_with_validity(array, dtype, new_validity)))
    }
}

impl CastKernel for OnPair {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }
        let new_validity = array.array().validity()?.cast_nullability(
            dtype.nullability(),
            array.array().len(),
            ctx,
        )?;
        Ok(Some(build_with_validity(array, dtype, new_validity)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(VarBinArray::from_iter(
        vec![Some("hello"), Some("world"), Some("hello world")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("foo"), None, Some("bar"), Some("foobar")],
        DType::Utf8(Nullability::Nullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("test")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    fn test_cast_onpair_conformance(#[case] array: VarBinArray) -> VortexResult<()> {
        let array = array.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let onpair = onpair_compress(&array, DEFAULT_DICT12_CONFIG, &mut ctx)?;
        test_cast_conformance(&onpair.into_array(), &mut ctx);
        Ok(())
    }
}
