// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Convert an [`OnPairArray`] to its canonical `VarBinViewArray` by running
//! the pure-Rust dictionary-lookup decoder over every row.

use std::sync::Arc;

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
use crate::OnPairArrayExt;
use crate::decode::OwnedDecodeInputs;

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
    let n = array.array().len();
    let lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    #[expect(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
        lengths.as_slice::<P>().iter().map(|x| *x as usize).sum()
    });

    let inputs = OwnedDecodeInputs::collect(array, ctx)?;
    let dv = inputs.view();
    // Fast path: `total_size` already known from `uncompressed_lengths`, so
    // skip the decoder's own size-precomputation pass. Single allocation,
    // single 4×-unrolled over-copy loop, no second scan.
    let mut buf: Vec<u8> = Vec::with_capacity(total_size + crate::MAX_TOKEN_SIZE);
    // SAFETY: capacity reserved above; `total_size` is the true decoded
    // byte count (sum of `uncompressed_lengths`).
    unsafe { dv.decode_rows_into_with_size(0, n, total_size, &mut buf) };
    let mut out_bytes = ByteBufferMut::with_capacity(buf.len());
    out_bytes.extend_from_slice(&buf);

    match_each_integer_ptype!(lengths.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            out_bytes,
            lengths.as_slice::<P>(),
        ))
    })
}
