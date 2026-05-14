// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;

/// Casts between Utf8/Binary that only differ in nullability are no-ops at
/// the bytes level: we rewrap the data into a new outer Array with the
/// requested DType.
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
                OnPair::new_unchecked_lazy(
                    dtype.clone(),
                    array.column_bytes_handle().clone(),
                    array.array().len(),
                    array.bits(),
                    array.dict_size(),
                    array.uncompressed_lengths().clone(),
                    new_validity,
                )
            }
            .into_array(),
        ))
    }
}

/// `CastKernel` and `CastReduce` are sibling traits in `vortex-array` — the
/// adaptor stack registers both — so we provide a forwarding kernel here.
impl CastKernel for OnPair {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        _ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as CastReduce>::cast(array, dtype)
    }
}
