// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteArray;
use arrow_array::types::ByteArrayType;
use vortex_compute::arrow::IntoArrow;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_byte_array<T: ByteArrayType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // If the Vortex array is already in VarBin format, we can directly convert it.
    if let Some(array) = array.as_opt::<VarBinVTable>() {
        return varbin_to_byte_array::<T>(array, ctx);
    }

    // Otherwise, we execute the array to a BinaryView vector and cast from there.
    let binary_view = array.execute(ctx)?.to_vector(ctx)?.into_arrow()?;
    arrow_cast::cast(&binary_view, &T::DATA_TYPE).map_err(VortexError::from)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
fn varbin_to_byte_array<T: ByteArrayType>(
    array: &VarBinArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // We must cast the offsets to the required offset type.
    let offsets = array
        .offsets()
        .cast(DType::Primitive(T::Offset::PTYPE, Nullability::NonNullable))?
        .execute(ctx)?
        .into_primitive()
        .buffer::<T::Offset>()
        .into_arrow_offset_buffer();

    let data = array.bytes().clone().into_arrow_buffer();

    let null_buffer = to_arrow_null_buffer(array.validity(), array.len(), ctx)?;
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets, data, null_buffer)
    }))
}
