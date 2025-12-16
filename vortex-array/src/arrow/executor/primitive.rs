// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::PrimitiveArray;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

pub(super) fn to_arrow_primitive<T: ArrowPrimitiveType>(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    // We use nullable here so we can essentially ignore nullability during the cast.
    let array = array.cast(DType::Primitive(T::Native::PTYPE, Nullability::Nullable))?;
    let vector = array.execute_vector(session)?.into_primitive();
    let (buffer, validity) = vector.downcast::<T::Native>().into_parts();
    let null_buffer = to_null_buffer(validity);
    let buffer = buffer.into_arrow_scalar_buffer();
    Ok(Arc::new(PrimitiveArray::<T>::new(buffer, null_buffer)))
}
