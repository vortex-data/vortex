// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Convert an [`OnPairArray`] to its canonical `VarBinViewArray` by handing
//! the materialised parts to `onpair::decompress_into`.

use std::sync::Arc;

use num_traits::AsPrimitive;
use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

pub(super) fn canonicalize_onpair(
    array: ArrayView<'_, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (buffers, views) = onpair_decode_views(array, 0, ctx)?;
    let validity = array.array().validity()?;
    Ok(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), array.dtype().clone(), validity)
            .into_array()
    })
}

pub(crate) fn onpair_decode_views(
    array: ArrayView<'_, OnPair>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    let total_size: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
        lengths
            .as_slice::<P>()
            .iter()
            .map(|&l| AsPrimitive::<usize>::as_(l))
            .sum()
    });

    // `codes_offsets` holds the per-row code boundaries and may itself be a
    // sliced or filtered view of the original. Its first and last entries
    // bound the contiguous run of `codes` belonging to the rows present in
    // this array: `slice` keeps the full `codes` child and only narrows
    // `codes_offsets` (so `code_start > 0` and/or `code_end < codes.len()`),
    // while `filter` rebuilds both children so the window is the whole stream.
    // OnPair has no `TakeExecute`, so a reordering take is served from the
    // canonical `VarBinView` and never reaches this path. We only need those
    // two boundaries, so point-look them up rather than decoding every offset.
    let codes_offsets = array.codes_offsets();
    let code_start = code_boundary_at(codes_offsets, 0, ctx)?;
    let code_end = code_boundary_at(codes_offsets, array.len(), ctx)?;
    vortex_ensure!(
        code_start <= code_end,
        "OnPair codes_offsets must be nondecreasing"
    );
    vortex_ensure!(
        code_end <= array.codes().len(),
        "OnPair codes_offsets end {} exceeds codes len {}",
        code_end,
        array.codes().len()
    );

    // Slice the `codes` child to that window *before* unpacking it, so a sliced
    // array materialises only its own codes rather than the whole column's. The
    // contiguous decoder walks `codes` in order and never reads the per-row
    // boundaries, so an empty boundary slice is sound.
    let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
    let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;

    let mut out_bytes = ByteBufferMut::with_capacity(total_size);
    let written = onpair::decompress_into(
        Parts {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            bits: array.bits(),
            codes: codes.as_slice(),
        },
        out_bytes.spare_capacity_mut(),
    );
    debug_assert_eq!(written, total_size);
    // SAFETY: `decompress_into` initialised exactly `written` bytes of the
    // spare capacity reserved above.
    unsafe { out_bytes.set_len(written) };

    match_each_integer_ptype!(lengths.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            out_bytes,
            lengths.as_slice::<P>(),
        ))
    })
}
