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
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_byte_array<T: ByteArrayType>(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // If the Vortex array is already in VarBin format, we can directly convert it.
    if let Some(array) = array.as_opt::<VarBinVTable>() {
        return varbin_to_byte_array::<T>(array, session);
    }

    // Otherwise, we execute the array to a BinaryView vector and cast from there.
    let binary_view = array.execute_vector(session)?.into_arrow()?;
    arrow_cast::cast(&binary_view, &T::DATA_TYPE).map_err(VortexError::from)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
fn varbin_to_byte_array<T: ByteArrayType>(
    array: &VarBinArray,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // We must cast the offsets to the required offset type.
    let offsets = array
        .offsets()
        .cast(DType::Primitive(T::Offset::PTYPE, Nullability::NonNullable))?
        .execute_vector(session)?
        .into_primitive()
        .downcast::<T::Offset>()
        .into_nonnull_buffer()
        .into_arrow_offset_buffer();

    let data = array.bytes().clone().into_arrow_buffer();

    let null_buffer = to_arrow_null_buffer(array.validity(), array.len(), session)?;
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets, data, null_buffer)
    }))
}
