// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;

/// Cast between `Utf8` and `Binary` (or adjust nullability) without touching
/// any of the encoded payload — we only rewrap into a new outer DType.
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
        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    dtype.clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets_handle().clone(),
                    array.codes_handle().clone(),
                    array.codes_offsets_handle().clone(),
                    array.uncompressed_lengths().clone(),
                    new_validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}

impl CastKernel for OnPair {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as CastReduce>::cast(array, dtype)
    }
}
