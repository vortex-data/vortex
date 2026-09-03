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
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::array::dict_view;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

pub(super) fn canonicalize_onpair(
    array: ArrayView<'_, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (out_bytes, lengths) = onpair_decode_bytes(array, ctx)?;
    let (buffers, views) = match_each_integer_ptype!(lengths.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            out_bytes.freeze(),
            lengths.as_slice::<P>(),
        )
    });
    let validity = array.array().validity()?;
    Ok(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), array.dtype().clone(), validity)
            .into_array()
    })
}

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
        let dict = dict_view(array, ctx)?;

        Ok(Self {
            codes,
            dict,
            lengths,
            total_size,
        })
    }

    /// Bulk-decodes the whole code stream into `out`, which must hold at least `total_size` bytes.
    ///
    /// Do not reach for `#[inline(always)]` here. `try_decode_into` is monomorphized for
    /// `CompactDictionaryView` whichever way this is annotated, and it is far too large to inline
    /// either way — it stays an out-of-line call in both. All `always` buys is dropping this
    /// wrapper's own frame, one `bl`/`ret` per decoded chunk against a whole-column decode, which
    /// `benches/decode.rs::canonicalize_to_varbinview` cannot tell apart from noise.
    #[inline]
    pub(crate) fn decode_into(&self, out: &mut [MaybeUninit<u8>]) -> VortexResult<usize> {
        let written = match onpair::try_decode_into(self.codes.as_slice(), self.dict, out) {
            Ok(written) => written,
            Err(_) => {
                vortex_bail!("OnPair codes decode to more bytes than uncompressed_lengths records")
            }
        };

        vortex_ensure!(
            written == self.total_size,
            "OnPair codes decoded to {written} bytes but uncompressed_lengths records {}",
            self.total_size
        );
        Ok(written)
    }
}

pub(crate) fn onpair_decode_bytes(
    array: ArrayView<'_, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBufferMut, PrimitiveArray)> {
    let plan = OnPairDecodePlan::new(array, ctx)?;
    let mut out_bytes = ByteBufferMut::with_capacity(plan.total_size);
    let written = plan.decode_into(out_bytes.spare_capacity_mut())?;
    // SAFETY: `decode_into` initialised exactly `written` bytes.
    unsafe { out_bytes.set_len(written) };
    Ok((out_bytes, plan.lengths))
}
