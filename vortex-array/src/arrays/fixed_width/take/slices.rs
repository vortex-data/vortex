// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::UnsignedPType;

pub(super) fn take_slices<T: Copy, S: UnsignedPType, L: UnsignedPType>(
    values: &Buffer<T>,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>> {
    let slices = starts
        .iter()
        .zip_eq(lengths)
        .map(|(&start, &length)| (start.as_(), length.as_()));
    copy_slices(values, slices, output_len, allocator)
}

pub(super) fn take_slices_constant_length<T: Copy, S: UnsignedPType>(
    values: &Buffer<T>,
    starts: &[S],
    length: usize,
    output_len: usize,
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>> {
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
        starts.iter().map(|start| (start.as_(), length)),
        output_len,
        allocator,
    )
}

// Keeping this kernel separate improves the take_fsl benchmark's small-range copies.
#[inline(never)]
fn copy_slices<T: Copy>(
    values: &Buffer<T>,
    slices: impl IntoIterator<Item = (usize, usize)>,
    output_len: usize,
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>> {
    output_len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    let mut result =
        BufferMut::<T>::with_capacity_aligned_in(output_len, values.alignment(), allocator.clone());
    let spare = &mut result.spare_capacity_mut()[..output_len];
    let mut cursor = 0usize;
    let record_count = values.len();

    for (start, length) in slices {
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray slice end overflows usize"))?;
        vortex_ensure!(
            end <= record_count,
            "PiecewiseSequenceArray slice {start}..{end} exceeds array length {record_count}"
        );
        let source = &values[start..end];
        spare[cursor..][..length].write_copy_of_slice(source);
        cursor += length;
    }

    // SAFETY: The loop initialized the prefix `0..cursor` of the spare capacity.
    unsafe { result.set_len(cursor) };
    vortex_ensure!(
        result.len() == output_len,
        "PiecewiseSequenceArray expanded length {} does not match declared length {output_len}",
        result.len()
    );
    Ok(result.freeze())
}
