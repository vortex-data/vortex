// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteViewArray;
use arrow_array::types::ByteViewType;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::dtype::from_arrow_data_type;
use crate::null_buffer::to_null_buffer;

/// Convert a canonical VarBinViewArray directly to Arrow.
pub fn canonical_varbinview_to_arrow<T: ByteViewType>(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let views = Buffer::<u128>::from_byte_buffer(array.views_handle().as_host().clone())
        .into_arrow_scalar_buffer();
    let buffers: Vec<_> = array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().clone().into_arrow_buffer())
        .collect();
    let nulls = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?,
    );

    // SAFETY: our own VarBinView array is considered safe.
    Ok(Arc::new(unsafe {
        GenericByteViewArray::<T>::new_unchecked(views, buffers, nulls)
    }))
}

pub fn execute_varbinview_to_arrow<T: ByteViewType>(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let compacted = array.compact_buffers(ctx)?;
    canonical_varbinview_to_arrow::<T>(&compacted, ctx)
}

pub(super) fn to_arrow_byte_view<T: ByteViewType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the array into the desired ByteView type.
    // We do this in case the vortex array is Utf8, and we want Binary or vice versa. By casting
    // first, we may push this down through the Vortex array tree. We choose nullable to be most
    // flexible since there's no prescribed nullability in Arrow types.
    let array = array.cast(from_arrow_data_type(&T::DATA_TYPE, Nullability::Nullable)?)?;

    let array = array.execute::<ArrayRef>(ctx)?;
    let varbinview = array.execute::<VarBinViewArray>(ctx)?;
    execute_varbinview_to_arrow::<T>(&varbinview, ctx)
}

#[cfg(test)]
mod tests {
    use arrow_array::types::StringViewType;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;

    use super::*;

    #[test]
    fn empty_views_are_aligned() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(std::iter::empty::<&str>());
        let mut ctx = array_session().create_execution_ctx();

        let arrow = canonical_varbinview_to_arrow::<StringViewType>(&array, &mut ctx)?;

        assert!(arrow.is_empty());
        Ok(())
    }
}
