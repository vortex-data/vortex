// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Index encoding for concatenated sequential ranges.
//!
//! A `PiecewiseSequentialArray` represents the expanded index sequence
//! `starts[i]..starts[i] + lengths[i]` for each piece `i`. It is intended for take operations that
//! can gather contiguous runs without materializing one index per element.

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::arrays::PrimitiveArray;
use crate::dtype::UnsignedPType;

pub mod array;
mod vtable;

#[cfg(test)]
mod tests;

pub use array::PiecewiseSequentialArrayExt;
pub use vtable::*;

pub(crate) fn check_index_arrays(starts: &ArrayRef, lengths: &ArrayRef) -> VortexResult<()> {
    check_index_array("starts", starts)?;
    check_index_array("lengths", lengths)?;
    vortex_ensure!(
        starts.len() == lengths.len(),
        "PiecewiseSequentialArray starts length {} does not match lengths length {}",
        starts.len(),
        lengths.len()
    );
    Ok(())
}

fn check_index_array(name: &str, array: &ArrayRef) -> VortexResult<()> {
    vortex_ensure!(
        array.dtype().is_unsigned_int(),
        "PiecewiseSequentialArray {name} must have unsigned integer dtype, got {}",
        array.dtype()
    );
    vortex_ensure!(
        !array.dtype().is_nullable(),
        "PiecewiseSequentialArray {name} must be non-nullable, got {}",
        array.dtype()
    );
    Ok(())
}

#[inline]
pub(crate) fn index_value_to_usize<T: UnsignedPType>(value: T) -> usize {
    value.as_()
}

#[inline]
pub(crate) fn index_value_to_u64<T: UnsignedPType + AsPrimitive<u64>>(value: T) -> u64 {
    value.as_()
}

#[inline]
pub(crate) fn checked_range_end(start: u64, length: usize) -> VortexResult<u64> {
    start
        .checked_add(length as u64)
        .ok_or_else(|| vortex_error::vortex_err!("PiecewiseSequentialArray range overflows u64"))
}

pub(crate) fn materialize_ranges<S, L>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    output_len: usize,
) -> VortexResult<BufferMut<u64>>
where
    S: UnsignedPType + AsPrimitive<u64>,
    L: UnsignedPType,
{
    let starts = starts.as_slice::<S>();
    let lengths = lengths.as_slice::<L>();
    let mut values = BufferMut::with_capacity(output_len);
    let mut computed_len = 0usize;

    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_u64(start);
        let length = index_value_to_usize(length);
        checked_range_end(start, length)?;
        computed_len = computed_len.checked_add(length).ok_or_else(|| {
            vortex_error::vortex_err!("PiecewiseSequentialArray output length overflows usize")
        })?;

        values.extend((0..length).map(|offset| start + offset as u64));
    }

    if computed_len != output_len {
        vortex_bail!(
            "PiecewiseSequentialArray expanded length {computed_len} does not match declared length {output_len}"
        );
    }
    Ok(values)
}
