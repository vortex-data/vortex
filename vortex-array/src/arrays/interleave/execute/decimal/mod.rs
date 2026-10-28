// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for decimal [`Interleave`](super::super::Interleave) values.
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
use crate::match_each_unsigned_integer_ptype;
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
    let allocator = ctx.allocator().clone();
    let values = (0..array.num_values())
        .map(|i| {
            let value = array.value(i);
            let len = value.len();
            let (data, row_mask) = if let Some(constant) = value.as_opt::<Constant>() {
                let payload = match constant.scalar().as_decimal().decimal_value() {
                    Some(value) => value.cast::<T>().ok_or_else(|| {
                        vortex_err!("interleave decimal constant must fit {dtype}, got {value}")
                    })?,
                    None => T::default(),
                };
                (Buffer::full_in(payload, 1, allocator.clone()), 0)
            } else {
                let decimal = value.as_::<Decimal>();
                let data = if decimal.values_type() == T::DECIMAL_TYPE {
                    decimal.buffer::<T>()
                } else {
                    // A decimal source can have noncanonical storage. Normalize before indexing
                    // rows so that every source has the same physical width. A narrowing copy
                    // checks the precision and zeros arbitrary null payloads.
                    let mask = decimal.validity()?.execute_mask(len, ctx)?;
                    let copy = match_each_decimal_value_type!(decimal.values_type(), |S| {
                        copy_to_canonical_width(decimal.buffer::<S>().as_slice(), &mask, dtype)?
                    });
                    Buffer::<T>::from_byte_buffer(copy.unwrap_host())
                };
                (data, usize::MAX)
            };

            Ok(Source {
                data,
                len,
                row_mask,
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
            gather(
                &values,
                branches.as_slice::<A>(),
                rows.as_slice::<R>(),
                &allocator,
            )
        })
    })
}

#[cfg(test)]
mod tests;
