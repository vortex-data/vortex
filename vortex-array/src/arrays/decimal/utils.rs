// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert native decimal buffers or narrow their integer children.
//!
//! Buffer conversions require materialized storage and retain unscaled values. Narrowing keeps
//! the precision-derived logical child dtype and selects storage from non-null bounds.

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::DecimalArray;
use crate::arrays::NarrowArray;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;

/// Return the array's unscaled values widened to `W`, which must be at least as wide as the
/// array's storage type.
pub(crate) fn widened_buffer<W: NativeDecimalType>(array: &DecimalArray) -> Buffer<W> {
    if array.values_type() == W::DECIMAL_TYPE {
        return array.buffer::<W>();
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        array
            .buffer::<T>()
            .iter()
            .map(|v| W::from(*v).vortex_expect("widening decimal cast must succeed"))
            .collect()
    })
}

/// Return the array's unscaled values converted to exactly `W`, whatever the array's
/// storage type. Zero-copy when the array is already stored at `W`.
///
/// Call [`DecimalArray::materialize_values`] before borrowing this buffer.
///
/// Widening is lossless. Narrowing fails for any _valid_ value that does not fit `W`.
/// Null slots may hold arbitrary bytes and never fail; their contents in the returned
/// buffer are likewise arbitrary and must not be read.
pub fn converted_buffer<W: NativeDecimalType>(
    array: &DecimalArray,
    validity: &Mask,
) -> VortexResult<Buffer<W>> {
    // Widening can never fail, so it needs no validation pass.
    if array.values_type() <= W::DECIMAL_TYPE {
        return Ok(widened_buffer(array));
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        let src = array.buffer::<T>();
        match validity {
            Mask::AllTrue(_) => {
                // Keeping the overflow scan branchless and vectorizable. Only on overflow
                // do we rescan for diagnostics.
                let any_overflow = src.iter().fold(false, |acc, v| acc | W::from(*v).is_none());
                if any_overflow {
                    let (i, v) = src
                        .iter()
                        .enumerate()
                        .find(|&(_, v)| W::from(*v).is_none())
                        .vortex_expect("overflow scan found an overflowing value");
                    vortex_bail!(
                        "decimal value {v} at index {i} does not fit {}",
                        W::DECIMAL_TYPE
                    );
                }
            }
            Mask::AllFalse(_) => return Ok(Buffer::zeroed(src.len())),
            Mask::Values(values) => {
                let any_overflow = src.iter().fold(false, |acc, v| acc | W::from(*v).is_none());
                if any_overflow {
                    for (i, v) in src.iter().enumerate() {
                        if values.value(i) && W::from(*v).is_none() {
                            vortex_bail!(
                                "decimal value {v} at index {i} does not fit {}",
                                W::DECIMAL_TYPE
                            );
                        }
                    }
                }
            }
        }
        // The convert pass is infallible: every valid value fits, and out-of-range null-slot
        // garbage is mapped to the default value. Callers must still ignore null slots.
        Ok(src
            .iter()
            .map(|v| W::from(*v).unwrap_or_default())
            .collect())
    })
}

/// Stores decimal values in a narrower integer child when all non-null values fit.
pub fn narrowed_decimal(
    decimal_array: DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalArray> {
    let values = NarrowArray::encode_signed(decimal_array.values().clone(), ctx)?;
    DecimalArray::try_new_values(values, decimal_array.decimal_dtype())
}
