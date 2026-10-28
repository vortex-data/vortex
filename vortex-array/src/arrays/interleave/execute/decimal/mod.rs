// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for decimal [`Interleave`] values.
//!
//! Sources can use different stored widths for the same logical dtype. Decimal execution
//! normalizes them to the logical canonical width before gathering rows.

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::super::InterleaveSlots;
use super::fixed_width::Source;
use super::fixed_width::gather;
use crate::AnyColumnar;
use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::copy_to_canonical_width;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_decimal_value_type;
use crate::require_child;

pub(super) fn execute(
    mut array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    array =
        require_child!(array, array.array_indices(), InterleaveSlots::ARRAY_INDICES => Primitive);
    array = require_child!(array, array.row_indices(), InterleaveSlots::ROW_INDICES => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), InterleaveSlots::VALUES_OFFSET + i => AnyColumnar);
    }

    let validity = array.as_ref().validity()?;
    let dtype = DecimalDType::try_from(array.dtype())?;
    let output = match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
        DecimalArray::new(gather_values::<T>(&array, dtype, ctx)?, dtype, validity)
    });

    Ok(ExecutionResult::done(output))
}

fn gather_values<T: NativeDecimalType>(
    array: &Array<Interleave>,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    let sources = (0..array.num_values())
        .map(|i| prepare_source::<T>(array.value(i), dtype, ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    gather(array, &sources, ctx.allocator())
}

/// Normalizes one decimal source while retaining a single payload for a constant.
fn prepare_source<T: NativeDecimalType>(
    value: &ArrayRef,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Source<T>> {
    if let Some(constant) = value.as_opt::<Constant>() {
        let payload = match constant.scalar().as_decimal().decimal_value() {
            Some(value) => value.cast::<T>().ok_or_else(|| {
                vortex_err!("interleave decimal constant must fit {dtype}, got {value}")
            })?,
            None => T::default(),
        };

        return Ok(Source::constant(payload, value.len(), ctx.allocator()));
    }

    let decimal = value.as_::<Decimal>();
    let data = if decimal.values_type() == T::DECIMAL_TYPE {
        decimal.buffer::<T>()
    } else {
        // A source can still store its values at a width other than the canonical width of its
        // dtype (#10701). The copy widens such values, or checks their precision and narrows them
        // with zeros in the null slots.
        let mask = decimal.validity()?.execute_mask(value.len(), ctx)?;
        let copy = match_each_decimal_value_type!(decimal.values_type(), |S| {
            copy_to_canonical_width(decimal.buffer::<S>().as_slice(), &mask, dtype)?
        });
        Buffer::<T>::from_byte_buffer(copy.unwrap_host())
    };

    Ok(Source::decoded(data))
}

#[cfg(test)]
mod tests;
