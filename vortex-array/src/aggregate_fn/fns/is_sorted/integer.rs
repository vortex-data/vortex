// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check signed integer ordering at the stored width.
//!
//! Decimal children and wide integer extensions share numeric ordering. Nulls sort first, and
//! strict ordering rejects repeated values, including repeated nulls.

use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedIteratorExt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::integer;
use crate::match_each_decimal_value_type;

pub(super) fn check_signed_integer_sorted(
    array: &ArrayRef,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let buffer = integer::materialize(array, ctx)?;
    let mask = buffer.validity.execute_mask(array.len(), ctx)?;
    match_each_decimal_value_type!(buffer.values_type, |T| {
        let values = Buffer::<T>::from_byte_buffer(buffer.values.to_host_sync());
        match mask {
            Mask::AllFalse(_) => Ok(!strict),
            Mask::AllTrue(_) => {
                let iter = values.iter().copied();

                Ok(if strict {
                    IsSortedIteratorExt::is_strict_sorted(iter)
                } else {
                    iter.is_sorted()
                })
            }
            Mask::Values(mask_values) => {
                let iter = mask_values
                    .bit_buffer()
                    .iter()
                    .zip_eq(values)
                    .map(|(is_valid, v)| is_valid.then_some(v));

                Ok(if strict {
                    IsSortedIteratorExt::is_strict_sorted(iter)
                } else {
                    iter.is_sorted()
                })
            }
        }
    })
}
