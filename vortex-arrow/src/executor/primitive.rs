// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_error::VortexResult;

use crate::null_buffer::to_null_buffer;

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
    // Arrow's physical primitive type is independent of field nullability. Preserve the
    // array's existing nullability so already-correct encoded arrays do not pay for a recursive
    // metadata-only cast before execution.
    let target_dtype = DType::Primitive(T::Native::PTYPE, array.dtype().nullability());
    let array = if array.dtype() == &target_dtype {
        array
    } else {
        array.cast(target_dtype)?
    };
    let primitive = array.execute::<PrimitiveArray>(ctx)?;
    canonical_primitive_to_arrow::<T>(primitive, ctx)
}
