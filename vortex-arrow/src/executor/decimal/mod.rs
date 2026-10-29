// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal export from stored values with the logical precision and scale.
//!
//! Exporting a `Widen` child directly avoids an intermediate widening copy. Narrowing conversions
//! ignore null payloads because those bytes do not constrain the Arrow value width.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Decimal32Array as ArrowDecimal32Array;
use arrow_array::Decimal64Array as ArrowDecimal64Array;
use arrow_array::Decimal128Array as ArrowDecimal128Array;
use arrow_array::Decimal256Array as ArrowDecimal256Array;
use arrow_buffer::i256;
use arrow_schema::DataType;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::null_buffer::to_null_buffer;

pub(super) fn to_arrow_decimal(
    array: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let decimal_dtype = *array.dtype().as_decimal_opt().ok_or_else(|| {
        vortex_err!(
            "Cannot convert Vortex array with dtype {} to an Arrow {} array",
            array.dtype(),
            data_type
        )
    })?;

    // A `Widen` holds the same values in a narrower child. Exporting the child copies each value
    // once, into the Arrow width. Executing the `Widen` first would also copy each value into the
    // canonical width of `decimal_dtype`.
    let array = array.execute_until::<Widen>(ctx)?;
    let values = match array.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => array,
    };
    let values = values.execute::<DecimalArray>(ctx)?;

    match data_type {
        DataType::Decimal32(..) => to_arrow_decimal32(values, decimal_dtype, ctx),
        DataType::Decimal64(..) => to_arrow_decimal64(values, decimal_dtype, ctx),
        DataType::Decimal128(..) => to_arrow_decimal128(values, decimal_dtype, ctx),
        DataType::Decimal256(..) => to_arrow_decimal256(values, decimal_dtype, ctx),
        _ => unreachable!("to_arrow_decimal called with non-decimal type"),
    }
}

/// Exports `values` as an Arrow `Decimal32` with the precision and scale of `decimal_dtype`.
///
/// `values` can be the narrower child of a [`Widen`], whose own decimal dtype has a smaller
/// precision. The Arrow precision and scale therefore come from `decimal_dtype`, never from
/// `values`.
fn to_arrow_decimal32(
    values: DecimalArray,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let validity = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;

    let buffer: Buffer<i32> = match values.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(values.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(values.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => values.buffer::<i32>(),
        DecimalType::I64 => narrow_values(values.buffer::<i64>(), &validity)?,
        DecimalType::I128 => narrow_values(values.buffer::<i128>(), &validity)?,
        DecimalType::I256 => {
            narrow_values(values.buffer::<vortex_array::dtype::i256>(), &validity)?
        }
    };

    Ok(Arc::new(
        ArrowDecimal32Array::new(buffer.into_arrow_scalar_buffer(), to_null_buffer(validity))
            .with_precision_and_scale(decimal_dtype.precision(), decimal_dtype.scale())?,
    ))
}

/// Like [`to_arrow_decimal32`], but exports an Arrow `Decimal64`.
fn to_arrow_decimal64(
    values: DecimalArray,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let validity = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;

    let buffer: Buffer<i64> = match values.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(values.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(values.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(values.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => values.buffer::<i64>(),
        DecimalType::I128 => narrow_values(values.buffer::<i128>(), &validity)?,
        DecimalType::I256 => {
            narrow_values(values.buffer::<vortex_array::dtype::i256>(), &validity)?
        }
    };

    Ok(Arc::new(
        ArrowDecimal64Array::new(buffer.into_arrow_scalar_buffer(), to_null_buffer(validity))
            .with_precision_and_scale(decimal_dtype.precision(), decimal_dtype.scale())?,
    ))
}

/// Like [`to_arrow_decimal32`], but exports an Arrow `Decimal128`.
fn to_arrow_decimal128(
    values: DecimalArray,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let validity = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;

    let buffer: Buffer<i128> = match values.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(values.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(values.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(values.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(values.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => values.buffer::<i128>(),
        DecimalType::I256 => {
            narrow_values(values.buffer::<vortex_array::dtype::i256>(), &validity)?
        }
    };

    Ok(Arc::new(
        ArrowDecimal128Array::new(buffer.into_arrow_scalar_buffer(), to_null_buffer(validity))
            .with_precision_and_scale(decimal_dtype.precision(), decimal_dtype.scale())?,
    ))
}

/// Like [`to_arrow_decimal32`], but exports an Arrow `Decimal256`.
fn to_arrow_decimal256(
    values: DecimalArray,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let validity = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;

    let buffer: Buffer<i256> = match values.values_type() {
        DecimalType::I8 => {
            Buffer::from_trusted_len_iter(values.buffer::<i8>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I16 => {
            Buffer::from_trusted_len_iter(values.buffer::<i16>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I32 => {
            Buffer::from_trusted_len_iter(values.buffer::<i32>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I64 => {
            Buffer::from_trusted_len_iter(values.buffer::<i64>().into_iter().map(|x| x.as_()))
        }
        DecimalType::I128 => Buffer::from_trusted_len_iter(
            values
                .buffer::<i128>()
                .into_iter()
                .map(|x| vortex_array::dtype::i256::from_i128(x).into()),
        ),
        DecimalType::I256 => {
            Buffer::<i256>::from_byte_buffer(values.buffer_handle().clone().into_host_sync())
        }
    };

    Ok(Arc::new(
        ArrowDecimal256Array::new(buffer.into_arrow_scalar_buffer(), to_null_buffer(validity))
            .with_precision_and_scale(decimal_dtype.precision(), decimal_dtype.scale())?,
    ))
}

/// Narrows the decimal `values` to `T`, and writes zero into every null row.
///
/// Every valid value **must** fit in `T`, or this returns an error. A null row can hold any value,
/// including one that does not fit in `T`, so this never converts it. The zeroed null rows match
/// [`copy_to_canonical_width`], but unlike it, this checks only that each valid value fits in `T`,
/// not that it fits the precision.
///
/// [`copy_to_canonical_width`]: vortex_array::arrays::decimal::copy_to_canonical_width
fn narrow_values<S: NativeDecimalType, T: NativeDecimalType>(
    values: Buffer<S>,
    validity: &Mask,
) -> VortexResult<Buffer<T>> {
    let narrow = |value: S| {
        <T as BigCast>::from(value).ok_or_else(|| {
            vortex_err!(
                "every valid {} value must fit in {} to narrow, got {value}",
                S::DECIMAL_TYPE,
                T::DECIMAL_TYPE,
            )
        })
    };

    match validity.bit_buffer() {
        AllOr::All => values
            .into_iter()
            .map(narrow)
            .process_results(|iter| Buffer::from_trusted_len_iter(iter)),
        AllOr::None => Ok(Buffer::zeroed(values.len())),
        AllOr::Some(is_valid) => values
            .into_iter()
            .zip(is_valid.iter())
            .map(|(value, is_valid)| {
                if is_valid {
                    narrow(value)
                } else {
                    Ok(T::default())
                }
            })
            .process_results(|iter| Buffer::from_trusted_len_iter(iter)),
    }
}

#[cfg(test)]
mod tests;
