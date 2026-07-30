// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Convert an [`OnPairArray`] to its canonical `VarBinViewArray` by handing
//! the materialised parts to `onpair::try_decode_into`.
//!
//! [`OnPairArray`]: crate::OnPairArray

use std::mem::MaybeUninit;
use std::sync::Arc;

use num_traits::AsPrimitive;
use onpair::CompactDictionaryView;
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
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::array::dict_view;
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

/// Everything needed to decode an OnPair array's values in one bulk `try_decode_into` call.
pub(crate) struct OnPairDecodePlan<'a> {
    codes: Buffer<u16>,
    dict: CompactDictionaryView<'a>,
    /// Per-row uncompressed lengths, zero for null rows.
    pub(crate) lengths: PrimitiveArray,
    /// Total decoded size, i.e. the sum of `lengths`.
    pub(crate) total_size: usize,
}

impl<'a> OnPairDecodePlan<'a> {
    pub(crate) fn new(array: ArrayView<'a, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
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

    // Slice the `codes` child to that window *before* unpacking it, so a sliced
    // array materialises only its own codes rather than the whole column's. The
    // contiguous decoder walks `codes` in order and never reads the per-row
    // boundaries, so an empty boundary slice is sound.
    let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
    let dict = dict_view(array, ctx)?;
    let mut out_bytes = ByteBufferMut::with_capacity(total_size);
    let written = onpair::try_decode_into(codes.as_slice(), dict, out_bytes.spare_capacity_mut())
        .map_err(|_| {
        vortex_err!("OnPair codes decode to more bytes than uncompressed_lengths records")
    })?;
    vortex_ensure!(
        written == total_size,
        "OnPair codes decoded to {written} bytes but uncompressed_lengths records {total_size}"
    );
    // SAFETY: `try_decode_into` initialised exactly `written` bytes.
    unsafe { out_bytes.set_len(written) };
    Ok((out_bytes, lengths))
}

pub(crate) fn onpair_decode_views(
    array: ArrayView<'_, OnPair>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let (out_bytes, lengths) = onpair_decode_bytes(array, ctx)?;
    match_each_integer_ptype!(lengths.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            out_bytes,
            lengths.as_slice::<P>(),
        ))
    })
}
