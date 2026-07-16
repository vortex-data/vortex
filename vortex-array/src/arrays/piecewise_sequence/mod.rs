// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Index encoding for concatenated sequential ranges.
//!
//! A `PiecewiseSequenceArray` represents the expanded index sequence
//! `starts[i] + j * multipliers[i]` for `j` in `0..lengths[i]` for each piece `i`. It is
//! intended for take operations that can gather regular runs without materializing one index per
//! element.

use itertools::Itertools;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::scalar::PValue;

pub mod array;
mod vtable;

#[cfg(test)]
mod tests;

pub use array::PiecewiseSequenceArrayExt;
pub use vtable::*;

pub(crate) fn check_index_arrays(
    starts: &ArrayRef,
    lengths: &ArrayRef,
    multipliers: &ArrayRef,
) -> VortexResult<()> {
    check_index_array("starts", starts)?;
    check_index_array("lengths", lengths)?;
    check_index_array("multipliers", multipliers)?;
    vortex_ensure!(
        starts.len() == lengths.len(),
        "PiecewiseSequenceArray starts length {} does not match lengths length {}",
        starts.len(),
        lengths.len()
    );
    vortex_ensure!(
        starts.len() == multipliers.len(),
        "PiecewiseSequenceArray starts length {} does not match multipliers length {}",
        starts.len(),
        multipliers.len()
    );
    Ok(())
}

pub(crate) fn execute_index_arrays(
    array: ArrayView<'_, PiecewiseSequence>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray, PrimitiveArray)> {
    let starts = array.starts().clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = array.lengths().clone().execute::<PrimitiveArray>(ctx)?;
    let multipliers = array.multipliers().clone().execute::<PrimitiveArray>(ctx)?;
    check_index_arrays(starts.as_ref(), lengths.as_ref(), multipliers.as_ref())?;
    Ok((starts, lengths, multipliers))
}

pub(crate) fn execute_unit_multiplier_index_arrays(
    array: ArrayView<'_, PiecewiseSequence>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(PrimitiveArray, PrimitiveArray)>> {
    if !is_constant_multiplier_one(array.multipliers()) {
        return Ok(None);
    }

    let starts = array.starts().clone().execute::<PrimitiveArray>(ctx)?;
    let lengths = array.lengths().clone().execute::<PrimitiveArray>(ctx)?;
    check_index_arrays(starts.as_ref(), lengths.as_ref(), array.multipliers())?;
    Ok(Some((starts, lengths)))
}

pub(crate) fn is_constant_multiplier_one(multipliers: &ArrayRef) -> bool {
    let Some(scalar) = multipliers.as_constant() else {
        return false;
    };
    matches!(
        scalar.as_primitive_opt().and_then(|scalar| scalar.pvalue()),
        Some(PValue::U8(1) | PValue::U16(1) | PValue::U32(1) | PValue::U64(1))
    )
}

fn check_index_array(name: &str, array: &ArrayRef) -> VortexResult<()> {
    vortex_ensure!(
        array.dtype().is_unsigned_int(),
        "PiecewiseSequenceArray {name} must have unsigned integer dtype, got {}",
        array.dtype()
    );
    vortex_ensure!(
        !array.dtype().is_nullable(),
        "PiecewiseSequenceArray {name} must be non-nullable, got {}",
        array.dtype()
    );
    Ok(())
}

pub(crate) fn materialize_ranges<S, L, M>(
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    multipliers: &PrimitiveArray,
    output_len: usize,
) -> VortexResult<BufferMut<u64>>
where
    S: UnsignedPType,
    L: UnsignedPType,
    M: UnsignedPType,
{
    let starts = starts.as_slice::<S>();
    let lengths = lengths.as_slice::<L>();
    let multipliers = multipliers.as_slice::<M>();
    let mut values = BufferMut::with_capacity(output_len);
    let mut computed_len = 0usize;

    for ((&start, &length), &multiplier) in starts.iter().zip_eq(lengths).zip_eq(multipliers) {
        let start: usize = start.as_();
        let length: usize = length.as_();
        let multiplier: usize = multiplier.as_();
        if length != 0 {
            let last_offset = length - 1;
            let last_delta = last_offset
                .checked_mul(multiplier)
                .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
            start
                .checked_add(last_delta)
                .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        }
        computed_len = computed_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;

        values.extend((0..length).map(|offset| (start + offset * multiplier) as u64));
    }

    if computed_len != output_len {
        vortex_bail!(
            "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
        );
    }
    Ok(values)
}

pub(crate) fn validate_index_ranges<S, L>(
    source_len: usize,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<()>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    let mut computed_len = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start: usize = start.as_();
        let length: usize = length.as_();
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        vortex_ensure!(
            end <= source_len,
            "PiecewiseSequenceArray range {start}..{end} exceeds source length {source_len}"
        );
        computed_len = computed_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    }

    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );
    Ok(())
}
