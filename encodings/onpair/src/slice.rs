// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Slicing an `OnPairArray` reuses the same dictionary blob and shares the
//! full `codes` byte buffer; we only narrow the per-row `codes_offsets`
//! window and adjust the validity / `uncompressed_lengths` children. No
//! decode, no re-training.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArrayExt;

impl SliceReduce for OnPair {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let codes_offsets =
            slice_codes_offsets(array.codes_offsets_bytes(), range.start, range.end)?;
        let uncompressed_lengths = array.uncompressed_lengths().slice(range.clone())?;
        let validity = array.array_validity().slice(range)?;
        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    array.dtype().clone(),
                    array.dict_bytes_handle().clone(),
                    array.dict_offsets_handle().clone(),
                    array.codes_handle().clone(),
                    codes_offsets,
                    uncompressed_lengths,
                    validity,
                    array.bits(),
                )
            }
            .into_array(),
        ))
    }
}

/// Slice the on-disk `codes_offsets` byte buffer to cover rows `[start, end)`.
/// Returns a new BufferHandle backed by a fresh `Buffer<u32>` of length
/// `end - start + 1`. We need the offsets themselves to stay byte-identical
/// (they index into the shared `codes` buffer), so this is a copy slice, not
/// a translate.
fn slice_codes_offsets(bytes: &ByteBuffer, start: usize, end: usize) -> VortexResult<BufferHandle> {
    let n_plus_one = end - start + 1;
    let byte_start = start * 4;
    let byte_end = byte_start + n_plus_one * 4;
    if byte_end > bytes.len() {
        return Err(vortex_err!(
            "OnPair slice: end {} exceeds codes_offsets bytes {}",
            byte_end,
            bytes.len()
        ));
    }
    let slice = bytes.as_slice();
    let mut out: Vec<u32> = Vec::with_capacity(n_plus_one);
    let mut i = byte_start;
    while i < byte_end {
        let arr: [u8; 4] = [slice[i], slice[i + 1], slice[i + 2], slice[i + 3]];
        out.push(u32::from_le_bytes(arr));
        i += 4;
    }
    Ok(BufferHandle::new_host(
        Buffer::<u32>::copy_from(out).into_byte_buffer(),
    ))
}
