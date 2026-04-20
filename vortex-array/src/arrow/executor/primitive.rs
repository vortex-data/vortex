// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;

/// Convert a canonical PrimitiveArray directly to Arrow.
pub fn canonical_primitive_to_arrow<T: ArrowPrimitiveType>(
    array: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Native: NativePType,
{
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    let null_buffer = to_null_buffer(validity);
    let buffer = array.into_buffer::<T::Native>().into_arrow_scalar_buffer();
    Ok(Arc::new(ArrowPrimitiveArray::<T>::new(buffer, null_buffer)))
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
    let primitive = array.execute::<PrimitiveArray>(ctx)?;
    canonical_primitive_to_arrow::<T>(primitive, ctx)
}
