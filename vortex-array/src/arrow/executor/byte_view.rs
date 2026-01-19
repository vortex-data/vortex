// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteViewArray;
use arrow_array::types::ByteViewType;
use arrow_buffer::ScalarBuffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

/// Convert a canonical VarBinViewArray directly to Arrow.
pub fn canonical_varbinview_to_arrow<T: ByteViewType>(array: &VarBinViewArray) -> ArrowArrayRef {
    let views =
        ScalarBuffer::<u128>::from(array.views().clone().into_byte_buffer().into_arrow_buffer());
    let buffers: Vec<_> = array
        .buffers()
        .iter()
        .map(|buffer| buffer.clone().into_arrow_buffer())
        .collect();
    let nulls = to_null_buffer(array.validity_mask());

    // SAFETY: our own VarBinView array is considered safe.
    Arc::new(unsafe { GenericByteViewArray::<T>::new_unchecked(views, buffers, nulls) })
}

pub fn execute_varbinview_to_arrow<T: ByteViewType>(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let views =
        ScalarBuffer::<u128>::from(array.views().clone().into_byte_buffer().into_arrow_buffer());
    let buffers: Vec<_> = array
        .buffers()
        .iter()
        .map(|buffer| buffer.clone().into_arrow_buffer())
        .collect();
    let nulls = to_arrow_null_buffer(array.validity(), array.len(), ctx)?;

    // SAFETY: our own VarBinView array is considered safe.
    Ok(Arc::new(unsafe {
        GenericByteViewArray::<T>::new_unchecked(views, buffers, nulls)
    }))
}

pub(super) fn to_arrow_byte_view<T: ByteViewType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the array into the desired ByteView type.
    // We do this in case the vortex array is Utf8, and we want Binary or vice versa. By casting
    // first, we may push this down through the Vortex array tree. We choose nullable to be most
    // flexible since there's no prescribed nullability in Arrow types.
    let array = array.cast(DType::from_arrow((&T::DATA_TYPE, Nullability::Nullable)))?;

    let varbinview = array.execute::<VarBinViewArray>(ctx)?;
    Ok(canonical_varbinview_to_arrow::<T>(&varbinview))
}
