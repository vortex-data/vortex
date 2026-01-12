// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::PrimitiveArray;
use arrow_array::types::DecimalType;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::DecimalDType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

// TODO(ngates): our i256 is different from Arrow's. Therefore we need an explicit `N` type
//  representing the Vortex native type that is equivalent to the Arrow native type.
pub(super) fn to_arrow_decimal<D: DecimalType, N: NativeDecimalType>(
    array: ArrayRef,
    precision: u8,
    scale: i8,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // Since Vortex doesn't have physical types, our DecimalDType only contains precision and scale.
    // When calling execute, Vortex may use any physical type >= the smallest type that can
    // hold the requested precision.
    //
    // We therefore create a fake precision that forces Vortex to use a native type that is at
    // least as wide as the requested Arrow type. We cast the array into this type.
    //
    // NOTE(ngates): we assume that a cast operation will produce the narrowest possible type that
    //  fits the requested precision.
    let fake_precision = precision.max(N::MAX_PRECISION);
    let array = array.cast(DType::Decimal(
        DecimalDType::new(fake_precision, scale),
        Nullability::Nullable,
    ))?;

    // Execute the array as a vector and downcast to our native type.
    let vector = array.execute(ctx)?.to_vector(ctx)?.into_decimal();
    vortex_ensure!(
        vector.decimal_type() == N::DECIMAL_TYPE,
        "Decimal array conversion produced unexpected decimal type: expected {:?}, got {:?}",
        N::DECIMAL_TYPE,
        vector.decimal_type()
    );

    let (_ps, buffer, validity) = N::downcast(vector).into_parts();
    let nulls = to_null_buffer(validity);

    assert_eq!(
        size_of::<D::Native>(),
        size_of::<N>(),
        "Mismatched native sizes between Arrow decimal type and Vortex native decimal type"
    );

    // SAFETY: we just checked that size_of::<D::Native> == size_of<N>. We also know that we have
    //  the same bit-representation as Arrow. We only need to transmute because we have different
    //  i256 types.
    let buffer = unsafe { std::mem::transmute::<Buffer<N>, Buffer<D::Native>>(buffer) };

    Ok(Arc::new(
        PrimitiveArray::<D>::new(buffer.into_arrow_scalar_buffer(), nulls)
            .with_precision_and_scale(precision, scale)
            .map_err(VortexError::from)?,
    ))
}
