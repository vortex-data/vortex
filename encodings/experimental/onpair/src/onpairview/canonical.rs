// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Canonicalise an [`OnPairViewArray`](crate::OnPairViewArray) to a
//! `VarBinViewArray`, choosing between three decode strategies.
//!
//! The [`onpair::decompress_into`] decoder walks a *contiguous* `codes` slice
//! sequentially and emits the decoded bytes; the per-row split is recovered
//! afterwards. So how we feed it depends on how the per-row windows are laid out
//! over the shared `codes` buffer (a freshly converted array is contiguous; a
//! `filter` leaves gaps; a reordering `take` scrambles the order):
//!
//! 1. **Span decode (no random access, may carry dead bytes).** When the windows
//!    are sorted and non-overlapping (contiguous, or contiguous-with-gaps after a
//!    `filter`), the *span* `codes[min_offset .. max_end]` already holds every
//!    live token in row order — plus, in the gap case, some dead tokens. We
//!    decode that whole span in **one sequential pass** and build `VarBinView`
//!    views that point at each row's byte range inside it
//!    ([`BinaryView::make_view`] takes an arbitrary buffer offset). The decoded
//!    gap bytes are never referenced — *dead values* in the output buffer. This
//!    is cache-friendly (no gather) and, when there are no gaps, optimal.
//!
//! 2. **Gather / compact.** When the windows are sorted but *sparse* (a very
//!    selective `filter` leaves large gaps), decoding the span would waste most
//!    of the work — and most of the output buffer — on dead bytes. Instead we
//!    gather the live windows into a fresh contiguous token buffer (random reads
//!    over `codes`) and decode only those. This is the same compaction the
//!    [`ListView`](vortex_array::arrays::ListViewArray) exporter performs before
//!    handing Arrow a sparse array.
//!
//! 3. **Gather is also the fallback** when the windows are *reordered or
//!    overlapping* (after a shuffling/duplicating `take`): a span decode can't
//!    reproduce row order then, so we always gather.
//!
//! `Auto` picks span-decode while the live fraction of the span is at least
//! [`SPAN_DECODE_DENSITY_THRESHOLD`] and gathers below it. The decision is an
//! `O(num_rows)` scan of the small per-row children, never `O(num_tokens)`.
//!
//! # What the benchmark says about "dead values"
//!
//! The intuition was that small gaps are fine — a span decode carrying a few
//! dead bytes avoids the random-access gather. The `view_compute` sweep shows
//! that for this decoder it does **not** pay off: gather is faster at *every*
//! gap density tested (5.3× at 2 % live, still 1.13× at 95 % live), and the
//! margin never closes. Two `O(span)` costs sink the span path whenever gaps
//! exist — decoding the dead tokens, and the prefix-sum over the span needed to
//! locate each row's start — and both are work the gather (which is `O(live)`)
//! simply doesn't do. The span path only ties/wins at *zero* gaps, where it
//! degenerates to a direct decode that skips the gather's copy entirely. So the
//! default threshold is `1.0`: take the no-copy direct decode iff fully
//! contiguous, otherwise gather. The threshold is left configurable, and
//! `SpanWithDead` remains selectable, for further experimentation.

use std::sync::Arc;

use num_traits::AsPrimitive;
use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::OnPairView;
use crate::OnPairViewArrayExt;
use crate::OnPairViewArraySlotsExt;
use crate::decode::collect_widened;

/// Minimum live fraction of the span (`Σ sizes / (max_end − min_offset)`) for
/// `Auto` to decode in place rather than gather.
///
/// Defaults to `1.0` — i.e. take the direct, no-copy span decode only when the
/// windows are fully contiguous, and gather whenever there is any gap. The
/// `view_compute` sweep shows gather beating the dead-carrying span decode at
/// every gap density (the span path pays two `O(span)` costs — decoding dead
/// tokens and a prefix-sum over the span — that gather avoids). Kept
/// configurable for experimentation.
pub const SPAN_DECODE_DENSITY_THRESHOLD: f32 = 1.0;

/// Which decode strategy to use when canonicalising an [`OnPairViewArray`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OnPairViewDecodeMode {
    /// Pick span-decode vs gather from the span density.
    Auto,
    /// Force the span decode (no gather); falls back to gather only if the
    /// windows are reordered/overlapping and a span decode is impossible.
    SpanWithDead,
    /// Force the gather/compact decode.
    Gather,
}

pub(super) fn canonicalize_onpairview(
    array: ArrayView<'_, OnPairView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    canonicalize_with(array, OnPairViewDecodeMode::Auto, ctx)
}

/// Canonicalise to a `VarBinViewArray` using an explicit [`OnPairViewDecodeMode`].
///
/// Exposed so callers (and benchmarks) can force a strategy; the VTable's
/// `execute` always uses [`OnPairViewDecodeMode::Auto`].
pub fn canonicalize_with(
    array: ArrayView<'_, OnPairView>,
    mode: OnPairViewDecodeMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (buffers, views) = onpairview_decode_views(array, 0, mode, ctx)?;
    let validity = array.array().validity()?;
    Ok(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), array.dtype().clone(), validity)
            .into_array()
    })
}

/// Layout summary of the per-row windows, from one `O(num_rows)` scan.
struct Layout {
    /// True if the non-empty windows are sorted ascending and non-overlapping,
    /// so a contiguous span decode reproduces row order.
    span_decodable: bool,
    /// Min start offset over non-empty windows.
    base: usize,
    /// Max end (`offset + size`) over non-empty windows.
    end: usize,
    /// Total live tokens (`Σ sizes`).
    live_tokens: usize,
}

impl Layout {
    fn span_tokens(&self) -> usize {
        self.end - self.base
    }

    /// Live fraction of the span: `1.0` when contiguous, →`0` as gaps grow.
    fn density(&self) -> f32 {
        let span = self.span_tokens();
        if span == 0 {
            1.0
        } else {
            self.live_tokens as f32 / span as f32
        }
    }
}

fn analyze(offsets: &[u32], sizes: &[u32]) -> Layout {
    debug_assert_eq!(offsets.len(), sizes.len());
    let mut span_decodable = true;
    let mut base = 0usize;
    let mut end = 0usize;
    let mut live_tokens = 0usize;
    let mut prev_end = 0usize;
    let mut started = false;

    for (&offset, &size) in offsets.iter().zip(sizes) {
        let size = size as usize;
        live_tokens += size;
        // Zero-length windows reference nothing, so they neither constrain the
        // span nor break ordering (their byte range is empty).
        if size == 0 {
            continue;
        }
        let offset = offset as usize;
        let window_end = offset + size;
        if !started {
            base = offset;
            end = window_end;
        } else {
            if offset < prev_end {
                span_decodable = false;
            }
            base = base.min(offset);
            end = end.max(window_end);
        }
        prev_end = window_end;
        started = true;
    }

    Layout {
        span_decodable,
        base,
        end,
        live_tokens,
    }
}

pub(crate) fn onpairview_decode_views(
    array: ArrayView<'_, OnPairView>,
    start_buf_index: u32,
    mode: OnPairViewDecodeMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let live_bytes: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
        lengths
            .as_slice::<P>()
            .iter()
            .map(|&l| AsPrimitive::<usize>::as_(l))
            .sum()
    });

    let offsets = array.collect_offsets(ctx)?;
    let sizes = array.collect_sizes(ctx)?;
    let layout = analyze(offsets.as_slice(), sizes.as_slice());

    let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;

    let use_span = layout.span_decodable
        && match mode {
            OnPairViewDecodeMode::Gather => false,
            OnPairViewDecodeMode::SpanWithDead => true,
            OnPairViewDecodeMode::Auto => layout.density() >= SPAN_DECODE_DENSITY_THRESHOLD,
        };

    // Span output exceeding a single buffer returns `None`; fall through to gather.
    if use_span
        && let Some(result) = decode_span_with_dead(
            array,
            &layout,
            offsets.as_slice(),
            &lengths,
            &dict_offsets,
            start_buf_index,
            ctx,
        )?
    {
        return Ok(result);
    }

    decode_gather(
        array,
        offsets.as_slice(),
        sizes.as_slice(),
        &lengths,
        &layout,
        live_bytes,
        &dict_offsets,
        start_buf_index,
        ctx,
    )
}

/// Strategy 1: decode the contiguous span `codes[base..end]` once and point each
/// row's view into it. Returns `None` if the decoded span would exceed a single
/// `VarBinView` buffer (caller then gathers).
#[allow(clippy::too_many_arguments)]
fn decode_span_with_dead(
    array: ArrayView<'_, OnPairView>,
    layout: &Layout,
    offsets: &[u32],
    lengths: &PrimitiveArray,
    dict_offsets: &Buffer<u32>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(Vec<ByteBuffer>, Buffer<BinaryView>)>> {
    let span_codes = collect_widened::<u16>(&array.codes().slice(layout.base..layout.end)?, ctx)?;

    // Prefix-sum of decoded bytes at each token boundary within the span, so we
    // can locate any row's start in the decoded output.
    let span_len = span_codes.len();
    let mut byte_at: Vec<usize> = Vec::with_capacity(span_len + 1);
    byte_at.push(0);
    let mut acc: usize = 0;
    for &code in span_codes.as_slice() {
        let code = code as usize;
        acc += (dict_offsets[code + 1] - dict_offsets[code]) as usize;
        byte_at.push(acc);
    }
    let decoded_span_bytes = acc;
    if decoded_span_bytes > MAX_BUFFER_LEN {
        return Ok(None);
    }

    let mut out = ByteBufferMut::with_capacity(decoded_span_bytes);
    let written = onpair::decompress_into(
        Parts {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            bits: array.bits(),
            codes: span_codes.as_slice(),
        },
        out.spare_capacity_mut(),
    );
    debug_assert_eq!(written, decoded_span_bytes);
    // SAFETY: `decompress_into` initialised exactly `written` bytes.
    unsafe { out.set_len(written) };
    let out = out.freeze();

    let mut views = BufferMut::<BinaryView>::with_capacity(offsets.len());
    match_each_integer_ptype!(lengths.ptype(), |P| {
        for (i, &offset) in offsets.iter().enumerate() {
            let len: usize = AsPrimitive::<usize>::as_(lengths.as_slice::<P>()[i]);
            if len == 0 {
                views.push(BinaryView::make_view(&[], start_buf_index, 0));
                continue;
            }
            let start = byte_at[offset as usize - layout.base];
            views.push(BinaryView::make_view(
                &out[start..start + len],
                start_buf_index,
                u32::try_from(start).map_err(|_| vortex_err!("span offset exceeds u32"))?,
            ));
        }
    });

    Ok(Some((vec![out], views.freeze())))
}

/// Decode the live windows into a single **compact** byte buffer in row order
/// (no dead values). Used by both the gather VarBinView path and the VarBin
/// export.
///
/// When the windows are already contiguous we slice `codes[base..end]` and
/// decode it directly (no gather copy, and we never materialise the whole
/// `codes` child); otherwise we gather the live windows first.
fn decode_compact_bytes(
    array: ArrayView<'_, OnPairView>,
    offsets: &[u32],
    sizes: &[u32],
    layout: &Layout,
    live_bytes: usize,
    dict_offsets: &Buffer<u32>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ByteBufferMut> {
    let contiguous = layout.span_decodable && layout.live_tokens == layout.span_tokens();
    let codes: Buffer<u16> = if contiguous {
        collect_widened::<u16>(&array.codes().slice(layout.base..layout.end)?, ctx)?
    } else {
        let all_codes = array.collect_codes(ctx)?;
        let mut gathered: Vec<u16> = Vec::with_capacity(layout.live_tokens);
        for (&offset, &size) in offsets.iter().zip(sizes) {
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
    };

    let mut out = ByteBufferMut::with_capacity(live_bytes);
    let written = onpair::decompress_into(
        Parts {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            bits: array.bits(),
            codes: codes.as_slice(),
        },
        out.spare_capacity_mut(),
    );
    debug_assert_eq!(written, live_bytes);
    // SAFETY: `decompress_into` initialised exactly `written` bytes.
    unsafe { out.set_len(written) };
    Ok(out)
}

/// Strategy 2/3: decode the live windows compactly and split into `VarBinView`
/// views sequentially.
#[allow(clippy::too_many_arguments)]
fn decode_gather(
    array: ArrayView<'_, OnPairView>,
    offsets: &[u32],
    sizes: &[u32],
    lengths: &PrimitiveArray,
    layout: &Layout,
    live_bytes: usize,
    dict_offsets: &Buffer<u32>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let out = decode_compact_bytes(array, offsets, sizes, layout, live_bytes, dict_offsets, ctx)?;
    match_each_integer_ptype!(lengths.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            out,
            lengths.as_slice::<P>(),
        ))
    })
}

/// Canonicalise an [`OnPairViewArray`](crate::OnPairViewArray) to a
/// **`VarBinArray`** (contiguous `bytes` + `offsets`) instead of a
/// `VarBinViewArray`.
///
/// Unlike a `VarBinView`, a `VarBin` cannot hold dead values, so this always
/// produces the compact form: the live windows are decoded into one contiguous
/// buffer (directly when contiguous, gathered otherwise) and `offsets` are the
/// running sum of the per-row lengths.
pub fn canonicalize_to_varbin(
    array: ArrayView<'_, OnPairView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let live_bytes: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
        lengths
            .as_slice::<P>()
            .iter()
            .map(|&l| AsPrimitive::<usize>::as_(l))
            .sum()
    });

    let offsets = array.collect_offsets(ctx)?;
    let sizes = array.collect_sizes(ctx)?;
    let layout = analyze(offsets.as_slice(), sizes.as_slice());
    let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;

    let bytes = decode_compact_bytes(
        array,
        offsets.as_slice(),
        sizes.as_slice(),
        &layout,
        live_bytes,
        &dict_offsets,
        ctx,
    )?;

    // Running-sum `offsets` (length `rows + 1`) over the per-row lengths.
    let mut varbin_offsets: BufferMut<i64> = BufferMut::with_capacity(lengths.len() + 1);
    varbin_offsets.push(0);
    let mut acc: i64 = 0;
    match_each_integer_ptype!(lengths.ptype(), |P| {
        for &len in lengths.as_slice::<P>() {
            acc += AsPrimitive::<i64>::as_(len);
            varbin_offsets.push(acc);
        }
    });

    let validity = array.array().validity()?;
    // SAFETY: `varbin_offsets` is a running sum of non-negative lengths, so it is
    // non-nullable, starts at 0, is monotonically non-decreasing, and its last
    // entry equals `bytes.len()`; the bytes were just decoded from a valid
    // `Utf8`/`Binary` column. So the `VarBinArray` invariants hold without the
    // (offset-scanning) validation `try_new` would run.
    Ok(unsafe {
        VarBinArray::new_unchecked(
            varbin_offsets.into_array(),
            bytes.freeze(),
            array.dtype().clone(),
            validity,
        )
    }
    .into_array())
}
