// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Canonicalise an [`OnPairViewArray`](crate::OnPairViewArray) to its canonical
//! `VarBinViewArray`.
//!
//! # Should we compact the `codes`?
//!
//! The [`onpair::decompress_into`] decoder walks `codes` *sequentially* and
//! writes decoded bytes; the per-row split is recovered afterwards from
//! `uncompressed_lengths`. So the decoder needs `codes` to be **exactly** the
//! concatenation of each row's token window, in row order, with nothing extra
//! in between. That gives two decode strategies:
//!
//! * **Direct slice (no copy).** If the per-row windows already form one
//!   contiguous, in-order run — `offsets` is sorted, `offsets[0] = base`, and
//!   `offsets[i] + sizes[i] == offsets[i + 1]` with no gaps — then
//!   `codes[base .. base + Σ sizes]` *is* that concatenation. We slice the
//!   shared `codes` child to exactly that window and decode it as-is. No gather,
//!   no allocation of a second token buffer. A freshly converted
//!   [`OnPair`](crate::OnPair) (or one that was only `slice`d) is always in this
//!   shape, so its canonicalisation is as cheap as [`OnPair`](crate::OnPair)'s.
//!
//! * **Gather / compact (one copy).** After a `filter` (drops rows → leaves
//!   *gaps* between surviving windows) or a reordering/overlapping `take`, the
//!   windows are no longer a clean contiguous run. Decoding `codes[first..last]`
//!   directly would feed the decoder the *gap* tokens too and corrupt the
//!   output, so we must first **compact** the surviving windows into a fresh
//!   contiguous `Vec<u16>` and decode that. This copies `Σ sizes` tokens — but
//!   those are precisely the tokens that have to be read to produce the output,
//!   so it is one extra pass over exactly the live data, never the whole column.
//!
//! We pick between them with an `O(num_rows)` scan of the (small) per-row
//! `offsets`/`sizes` children — never an `O(num_tokens)` scan — so detecting the
//! cheap path is itself cheap.
//!
//! ## When is the OnPairView route faster overall?
//!
//! Compared with the [`OnPair`](crate::OnPair) "compact on every op" pipeline —
//! where each `filter`/`take` rebuilds the surviving token stream — the
//! OnPairView route keeps `filter`/`take` metadata-only and pays a *single*
//! gather here at canonicalisation. So it wins when the token stream is rebuilt
//! more than once (a chain of filters/takes), when intermediate results are
//! large (low early selectivity), or when the result is never canonicalised at
//! all (the gather never happens). The compacting route only competes when there
//! is exactly one, highly selective op immediately followed by a decode — then
//! both copy roughly the same (few) survivors once.

use std::ops::Range;
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

use crate::OnPairView;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;
use crate::decode::collect_widened;

pub(super) fn canonicalize_onpairview(
    array: ArrayView<'_, OnPairView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (buffers, views) = onpairview_decode_views(array, 0, ctx)?;
    let validity = array.array().validity()?;
    Ok(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), array.dtype().clone(), validity)
            .into_array()
    })
}

/// If the per-row windows form one contiguous, in-order run, return the single
/// `codes` range that covers them all; otherwise `None` (a gather is required).
///
/// This is the "should we compact?" decision: `Some` ⇒ decode in place, `None`
/// ⇒ compact first. Runs in `O(num_rows)`.
fn contiguous_run(offsets: &[u32], sizes: &[u32]) -> Option<Range<usize>> {
    debug_assert_eq!(offsets.len(), sizes.len());
    let Some(&base) = offsets.first() else {
        return Some(0..0);
    };
    let mut expected = base as usize;
    for (&offset, &size) in offsets.iter().zip(sizes) {
        if offset as usize != expected {
            return None;
        }
        expected += size as usize;
    }
    Some(base as usize..expected)
}

pub(crate) fn onpairview_decode_views(
    array: ArrayView<'_, OnPairView>,
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

    // Cheap O(num_rows) scan of the small per-row children decides the strategy.
    let offsets = array.collect_offsets(ctx)?;
    let sizes = array.collect_sizes(ctx)?;

    let codes: Buffer<u16> = match contiguous_run(offsets.as_slice(), sizes.as_slice()) {
        // Direct slice: the windows are already a contiguous run, so we hand the
        // decoder exactly that slice of the shared `codes` child — no gather.
        Some(range) => {
            vortex_ensure!(
                range.end <= array.codes().len(),
                "OnPairView contiguous range {:?} exceeds codes len {}",
                range,
                array.codes().len()
            );
            collect_widened::<u16>(&array.codes().slice(range)?, ctx)?
        }
        // Gather / compact: copy the live windows (and only those) into a fresh
        // contiguous token buffer in row order.
        None => {
            let all_codes = array.collect_codes(ctx)?;
            let total_tokens: usize = sizes.as_slice().iter().map(|&s| s as usize).sum();
            let mut gathered: Vec<u16> = Vec::with_capacity(total_tokens);
            for (&offset, &size) in offsets.as_slice().iter().zip(sizes.as_slice()) {
                let offset = offset as usize;
                let end = offset + size as usize;
                vortex_ensure!(
                    end <= all_codes.len(),
                    "OnPairView window [{offset}, {end}) exceeds codes len {}",
                    all_codes.len()
                );
                gathered.extend_from_slice(&all_codes.as_slice()[offset..end]);
            }
            Buffer::from(gathered)
        }
    };

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
