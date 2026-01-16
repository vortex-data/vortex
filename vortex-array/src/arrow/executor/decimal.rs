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
use crate::arrays::DecimalArray;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

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

    // Execute the array as a DecimalArray and extract its components.
    let decimal_array = array.execute::<DecimalArray>(ctx)?;
    vortex_ensure!(
        decimal_array.values_type() == N::DECIMAL_TYPE,
        "Decimal array conversion produced unexpected decimal type: expected {:?}, got {:?}",
        N::DECIMAL_TYPE,
        decimal_array.values_type()
    );

    let buffer = decimal_array.buffer::<N>();
    let nulls = to_arrow_null_buffer(decimal_array.validity(), decimal_array.len(), ctx)?;

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
