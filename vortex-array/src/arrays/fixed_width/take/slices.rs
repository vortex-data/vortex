// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use itertools::Itertools as _;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::UnsignedPType;

pub(super) fn take_slices<S: UnsignedPType, L: UnsignedPType>(
    values: &ByteBuffer,
    byte_width: usize,
    record_count: usize,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<ByteBuffer> {
    let slices = starts
        .iter()
        .zip_eq(lengths)
        .map(|(&start, &length)| (start.as_(), length.as_()));
    copy_slices(values, byte_width, record_count, slices, output_len)
}

pub(super) fn take_slices_constant_length<S: UnsignedPType>(
    values: &ByteBuffer,
    byte_width: usize,
    record_count: usize,
    starts: &[S],
    length: usize,
    output_len: usize,
) -> VortexResult<ByteBuffer> {
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );
    copy_slices(
        values,
        byte_width,
        record_count,
        starts.iter().map(|start| (start.as_(), length)),
        output_len,
    )
}

fn copy_slices(
    values: &ByteBuffer,
    byte_width: usize,
    record_count: usize,
    slices: impl IntoIterator<Item = (usize, usize)>,
    output_len: usize,
) -> VortexResult<ByteBuffer> {
    let input_byte_len = record_count
        .checked_mul(byte_width)
        .ok_or_else(|| vortex_err!("Fixed-width values buffer length overflows usize"))?;
    vortex_ensure!(
        values.len() == input_byte_len,
        "Fixed-width values buffer length does not match record count"
    );

    let output_byte_len = output_len
        .checked_mul(byte_width)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    let mut result = BufferMut::<u8>::with_capacity_aligned(output_byte_len, values.alignment());
    let spare = &mut result.spare_capacity_mut()[..output_byte_len];
    let mut cursor = 0usize;

    for (start, length) in slices {
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray slice end overflows usize"))?;
        vortex_ensure!(
            end <= record_count,
            "PiecewiseSequenceArray slice {start}..{end} exceeds array length {record_count}"
        );
        // These multiplications cannot overflow because `end <= record_count` and the complete
        // values buffer length was checked above.
        let byte_start = start * byte_width;
        let byte_length = length * byte_width;
        let source = &values[byte_start..][..byte_length];
        // SAFETY: `source` and the checked spare-capacity range have equal lengths and do not
        // overlap.
        unsafe {
            ptr::copy_nonoverlapping(
                source.as_ptr(),
                spare[cursor..][..source.len()].as_mut_ptr().cast::<u8>(),
                source.len(),
            );
        }
        cursor += source.len();
    }

    // SAFETY: The loop initialized the prefix `0..cursor` of the spare capacity.
    unsafe { result.set_len(cursor) };
    vortex_ensure!(
        result.len() == output_byte_len,
        "PiecewiseSequenceArray expanded length {} does not match declared length {output_byte_len}",
        result.len()
    );
    Ok(result.freeze().into_byte_buffer())
}
