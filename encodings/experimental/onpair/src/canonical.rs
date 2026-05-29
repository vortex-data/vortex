// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Convert an [`OnPairArray`] to its canonical `VarBinViewArray` by handing
//! the materialised parts to `onpair::decompress_into`.

use std::sync::Arc;

use num_traits::AsPrimitive;
use onpair::DECOMPRESS_BUFFER_PADDING;
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

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::decode::FullDecodeInputs;

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

    let inputs = FullDecodeInputs::collect(array, ctx)?;

    let total_size: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
        lengths
            .as_slice::<P>()
            .iter()
            .map(|&l| AsPrimitive::<usize>::as_(l))
            .sum()
    });

    let mut out_bytes = ByteBufferMut::with_capacity(total_size + DECOMPRESS_BUFFER_PADDING);
    let written = inputs.decompress_into(out_bytes.spare_capacity_mut());
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
