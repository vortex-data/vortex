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
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

// TODO(ngates): our i256 is different from Arrow's. Therefore we need an explicit `N` type.
pub(super) fn to_arrow_decimal<D: DecimalType, N: NativeDecimalType>(
    array: ArrayRef,
    precision: u8,
    scale: i8,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    // Since Vortex doesn't have physical types, the vector execution will try to use the
    // narrowest type that fits the precision + scale.
    //
    // We therefore fake the required precision such that the correct native type is used.
    let fake_precision = precision.max(N::MAX_PRECISION);
    let array = array.cast(DType::Decimal(
        DecimalDType::new(fake_precision, scale),
        Nullability::Nullable,
    ))?;

    // First we cast the array into the desired Decimal type before executing into a vector.
    let vector = array.execute_vector(session)?.into_decimal();
    vortex_ensure!(
        vector.decimal_type() == N::DECIMAL_TYPE,
        "Decimal array conversion produced unexpected decimal type: expected {:?}, got {:?}",
        N::DECIMAL_TYPE,
        vector.decimal_type()
    );

    let (_ps, buffer, validity) = N::downcast(vector).into_parts();
    let nulls = to_null_buffer(validity);

    // Again, because our i256 type is different from Arrow's, we need to special-case it.
    let buffer = unsafe { std::mem::transmute::<Buffer<N>, Buffer<D::Native>>(buffer) };

    Ok(Arc::new(
        PrimitiveArray::<D>::new(buffer.into_arrow_scalar_buffer(), nulls)
            .with_precision_and_scale(precision, scale)
            .map_err(VortexError::from)?,
    ))
}
