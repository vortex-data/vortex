// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::PrimitiveArray;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

/// Convert a canonical PrimitiveArray directly to Arrow.
pub fn canonical_primitive_to_arrow<T: ArrowPrimitiveType>(array: PrimitiveArray) -> ArrowArrayRef
where
    T::Native: NativePType,
{
    let validity = array.validity_mask();
    let null_buffer = to_null_buffer(validity);
    let buffer = array.into_buffer::<T::Native>().into_arrow_scalar_buffer();
    Arc::new(ArrowPrimitiveArray::<T>::new(buffer, null_buffer))
}

pub(super) fn to_arrow_primitive<T: ArrowPrimitiveType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    // We use nullable here so we can essentially ignore nullability during the cast.
    let array = array.cast(DType::Primitive(T::Native::PTYPE, Nullability::Nullable))?;
    let primitive = array.execute(ctx)?.into_primitive();
    Ok(canonical_primitive_to_arrow::<T>(primitive))
}
