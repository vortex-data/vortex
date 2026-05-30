// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use super::array::FSSTView;
use super::array::FSSTViewArrayExt;
use super::array::FSSTViewArraySlotsExt;

/// Canonicalize an [`FSSTView`] array into a [`VarBinViewArray`].
///
/// Because `filter`/`take`/`slice` leave the compressed byte heap untouched, the live codes of
/// element `i` are the (possibly out-of-order, possibly overlapping) slice
/// `codes_bytes[offset_i .. offset_i + size_i]`. We first gather them into element order, then
/// bulk-decompress in a single pass and build the binary views from the uncompressed lengths.
pub(super) fn canonicalize_fsstview(
    array: ArrayView<'_, FSSTView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let bytes = array.codes_bytes();

    let offsets = array
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let sizes = array.codes_sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let uncompressed_lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    #[expect(clippy::cast_possible_truncation)]
    let offsets: Vec<usize> = match_each_integer_ptype!(offsets.ptype(), |O| {
        offsets
            .as_slice::<O>()
            .iter()
            .map(|o| *o as usize)
            .collect()
    });
    #[expect(clippy::cast_possible_truncation)]
    let sizes: Vec<usize> = match_each_integer_ptype!(sizes.ptype(), |S| {
        sizes.as_slice::<S>().iter().map(|s| *s as usize).collect()
    });

    // Gather the live compressed bytes into element order.
    let total_compressed: usize = sizes.iter().sum();
    let mut compressed = ByteBufferMut::with_capacity(total_compressed);
    for i in 0..len {
        compressed.extend_from_slice(&bytes[offsets[i]..offsets[i] + sizes[i]]);
    }

    #[expect(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(uncompressed_lengths.ptype(), |P| {
        uncompressed_lengths
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .sum()
    });

    // Bulk-decompress the gathered heap. We reserve 7 extra bytes because the FSST decoder may
    // overrun the output by up to a word.
    let decompressor = array.decompressor();
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + 7);
    let written = decompressor.decompress_into(
        compressed.as_slice(),
        uncompressed_bytes.spare_capacity_mut(),
    );
    unsafe { uncompressed_bytes.set_len(written) };

    let (buffers, views) = match_each_integer_ptype!(uncompressed_lengths.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            uncompressed_bytes,
            uncompressed_lengths.as_slice::<P>(),
        )
    });

    // SAFETY: FSST validates the bytes for binary/UTF-8; the views point at valid ranges.
    Ok(unsafe {
        VarBinViewArray::new_unchecked(
            views,
            Arc::from(buffers),
            array.dtype().clone(),
            array.fsstview_validity(),
        )
        .into_array()
    })
}
