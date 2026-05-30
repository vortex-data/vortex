// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalization of [`FSSTView`] into a [`VarBinViewArray`].
//!
//! After metadata-only `filter`/`take`, an [`FSSTView`]'s byte heap is the *original* heap and
//! the live codes are scattered (gaps after a filter, reordering/duplication after a take). To
//! canonicalize we must produce one contiguous decompressed buffer in element order. There are
//! three ways to get there, with different cost profiles — see [`FsstViewCompaction`]:
//!
//! - [`Direct`][FsstViewCompaction::Direct]: the live codes are still contiguous and in order
//!   (e.g. an untouched view or one that was only sliced). We bulk-decompress that single
//!   contiguous range with no copy. Fastest, but only valid when contiguous.
//! - [`GatherBulk`][FsstViewCompaction::GatherBulk] ("compact"): copy the scattered live codes
//!   into a contiguous buffer, then a *single* bulk decompress. Pays a copy of the live
//!   compressed bytes but the one bulk call amortizes the FSST 8-wide fast path across all
//!   element boundaries.
//! - [`PerElement`][FsstViewCompaction::PerElement] ("no compact"): decompress each element's
//!   slice directly into its place in the output. No copy, but one decompress call per element.
//! - [`RunCoalesce`][FsstViewCompaction::RunCoalesce] ("export paired slices"): decode contiguous
//!   heap runs straight into a *heap-ordered* output and point `VarBinView`s back into it, out of
//!   order — no gather copy, dedups duplicates.
//!
//! The compaction question, concretely. The `fsst_view_compute` benchmark (two ~2 MiB inputs,
//! ~12-byte and ~256-byte strings) shows **`GatherBulk` is the best non-contiguous strategy across
//! the whole tested range**, for both short and long strings. The reason FSST decode is shaped
//! this way: a fast 8-wide body and a slow byte-by-byte tail. `PerElement` pays that tail *once
//! per element* (N tails); `GatherBulk` decodes the whole heap in one call and pays it *once*,
//! which dominates the gather memcpy even at 256-byte strings.
//!
//! `RunCoalesce` was the appealing idea of skipping the gather entirely — decode runs in place and
//! let the `VarBinView` reference them out of order. It loses anyway, badly for short strings
//! (`take many_short/shuffle`: ~18 ms vs ~5.6 ms for `GatherBulk`). The reason is subtle: the
//! random access you avoid at *decode* time reappears at *view-build* time. Views are built in
//! element order, so over a heap-ordered output the per-element `make_view` does N cache-missing
//! random reads (and, for ≤12-byte strings, random-access *inlining* copies), plus an
//! O(N log N) sort. `GatherBulk`'s output is element-ordered, so its view-build is sequential. The
//! cheap sequential gather memcpy beats the expensive scattered view construction.
//!
//! So [`FsstViewCompaction::Auto`] uses `Direct` when the live codes are contiguous
//! (untouched/sliced view) and `GatherBulk` otherwise. `PerElement` and `RunCoalesce` are kept
//! selectable so the trade-off stays measurable, but `Auto` never picks them.

use std::sync::Arc;

use fsst::Decompressor;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use super::array::FSSTView;
use super::array::FSSTViewArrayExt;
use super::array::FSSTViewArraySlotsExt;

/// Strategy for materializing the decompressed bytes when canonicalizing an [`FSSTView`].
///
/// See the [module docs][self] for the full trade-off analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsstViewCompaction {
    /// Pick a strategy automatically: `Direct` when the live codes are contiguous, else
    /// `GatherBulk`. Never picks `PerElement` or `RunCoalesce` (both lose; see module docs).
    Auto,
    /// Bulk-decompress the contiguous live range with no copy. Falls back to `GatherBulk` if the
    /// view's codes are not contiguous and in order.
    Direct,
    /// Compact the scattered live codes into a contiguous buffer, then a single bulk decompress.
    GatherBulk,
    /// Decompress each element's code slice directly into place, without compacting.
    PerElement,
    /// Coalesce survivors into contiguous heap runs and decompress each run with a *single* call
    /// directly into a heap-ordered output (no gather copy), emitting `VarBinView` views — possibly
    /// out of order — that point back into it. Decodes distinct codes once (duplicates share a
    /// view).
    ///
    /// This is the "export paired slices into a `VarBinView`" approach. In theory it skips the
    /// gather copy entirely; in practice it loses to `GatherBulk` (see module docs) because the
    /// random access just moves to view-build time, where it's more expensive. Retained for
    /// measurement only — `Auto` never selects it.
    RunCoalesce,
}

pub(super) fn canonicalize_fsstview(
    array: ArrayView<'_, FSSTView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    canonicalize_fsstview_with(array, FsstViewCompaction::Auto, ctx)
}

/// Canonicalize an [`FSSTView`] to a [`VarBinViewArray`] using an explicit compaction strategy.
///
/// Exposed (rather than only the dispatch-driven [`canonicalize_fsstview`]) so benchmarks can
/// measure each strategy directly. Production code goes through [`FsstViewCompaction::Auto`].
pub fn canonicalize_fsstview_with(
    array: ArrayView<'_, FSSTView>,
    strategy: FsstViewCompaction,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets = load_usize(array.codes_offsets(), ctx)?;
    let sizes = load_usize(array.codes_sizes(), ctx)?;

    let ulen_prim = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    #[expect(clippy::cast_possible_truncation)]
    let ulens: Vec<usize> = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
        ulen_prim
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .collect()
    });
    let total_size: usize = ulens.iter().sum();
    let live: usize = sizes.iter().sum();

    let heap_buffer = array.codes_bytes();
    let heap = heap_buffer.as_slice();
    let decompressor = array.decompressor();

    let contiguous = is_contiguous(&offsets, &sizes);
    let chosen = match strategy {
        // Direct when the live codes are still one contiguous run, else compact-and-bulk.
        // `GatherBulk` beats both `PerElement` and `RunCoalesce` across the whole practical range
        // (see module docs), so `Auto` picks neither.
        FsstViewCompaction::Auto => {
            if contiguous {
                FsstViewCompaction::Direct
            } else {
                FsstViewCompaction::GatherBulk
            }
        }
        // `Direct` is only valid for a contiguous layout; fall back to a compacting decode.
        FsstViewCompaction::Direct if !contiguous => FsstViewCompaction::GatherBulk,
        other => other,
    };

    // RunCoalesce builds its own (buffers, views) — decompression order is decoupled from element
    // order, so it can't go through `build_views` (which assumes element-order contiguous output).
    if chosen == FsstViewCompaction::RunCoalesce {
        let (buffers, views) =
            decompress_run_coalesce(&decompressor, heap, &offsets, &sizes, &ulens, total_size);
        // SAFETY: FSST validates the bytes for binary/UTF-8; the views point at valid ranges.
        return Ok(unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from(buffers),
                array.dtype().clone(),
                array.fsstview_validity(),
            )
            .into_array()
        });
    }

    let uncompressed = match chosen {
        FsstViewCompaction::Direct => {
            let start = offsets.first().copied().unwrap_or(0);
            decompress_direct(&decompressor, heap, start, live, total_size)
        }
        FsstViewCompaction::GatherBulk => {
            decompress_gather(&decompressor, heap, &offsets, &sizes, live, total_size)
        }
        // `Auto`/`RunCoalesce` are resolved above.
        _ => decompress_per_element(&decompressor, heap, &offsets, &sizes, &ulens, total_size),
    };

    let (buffers, views) = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
        build_views(0, MAX_BUFFER_LEN, uncompressed, ulen_prim.as_slice::<P>())
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

fn load_usize(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let prim = array.clone().execute::<PrimitiveArray>(ctx)?;
    #[expect(clippy::cast_possible_truncation)]
    let out: Vec<usize> = match_each_integer_ptype!(prim.ptype(), |P| {
        prim.as_slice::<P>().iter().map(|x| *x as usize).collect()
    });
    Ok(out)
}

/// Returns true if the live codes occupy a single contiguous, in-order run of the heap.
fn is_contiguous(offsets: &[usize], sizes: &[usize]) -> bool {
    let Some(&first) = offsets.first() else {
        return true;
    };
    let mut pos = first;
    for (&offset, &size) in offsets.iter().zip(sizes) {
        if offset != pos {
            return false;
        }
        pos += size;
    }
    true
}

/// Decompress a single contiguous run of the heap in one bulk call (no copy).
fn decompress_direct(
    decompressor: &Decompressor<'_>,
    heap: &[u8],
    start: usize,
    live: usize,
    total_size: usize,
) -> ByteBufferMut {
    let mut out = ByteBufferMut::with_capacity(total_size + 7);
    let written =
        decompressor.decompress_into(&heap[start..start + live], out.spare_capacity_mut());
    unsafe { out.set_len(written) };
    out
}

/// Compact the scattered live codes into a contiguous buffer, then a single bulk decompress.
fn decompress_gather(
    decompressor: &Decompressor<'_>,
    heap: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    live: usize,
    total_size: usize,
) -> ByteBufferMut {
    let mut compressed = ByteBufferMut::with_capacity(live);
    for (&offset, &size) in offsets.iter().zip(sizes) {
        compressed.extend_from_slice(&heap[offset..offset + size]);
    }
    let mut out = ByteBufferMut::with_capacity(total_size + 7);
    let written = decompressor.decompress_into(compressed.as_slice(), out.spare_capacity_mut());
    unsafe { out.set_len(written) };
    out
}

/// Coalesce survivors into contiguous heap runs, decompress each run once directly into the
/// output, and build `VarBinView`s (in element order) pointing back into that output.
///
/// Distinct elements are keyed by their `(offset, size)` heap span: duplicates (from a `take`
/// with repeats) are decoded once and share a view. Adjacent distinct spans (`offset == prev end`)
/// are decompressed in a single FSST call, so a shuffle take of the whole array is one decode.
fn decompress_run_coalesce(
    decompressor: &Decompressor<'_>,
    heap: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    ulens: &[usize],
    total_size: usize,
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let count = offsets.len();

    // Visit elements in heap order. Sorting by `(offset, size)` groups duplicates (same span)
    // together and, at a shared offset, orders the zero-size span (null/empty) before the
    // non-zero one — keeping the run extension below well-defined. `size` is part of the key
    // because a zero-size element shares an offset with its heap neighbour.
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_unstable_by_key(|&i| (offsets[i], sizes[i]));

    // Output position of each element's decoded bytes, filled below.
    let mut out_pos = vec![0usize; count];
    let mut out = ByteBufferMut::with_capacity(total_size + 7);
    let spare = out.spare_capacity_mut();

    let mut written = 0usize;
    let mut cursor = 0usize;
    while cursor < count {
        let head = order[cursor];
        // Zero-size spans (empty/null) decode to nothing; share the current position.
        if sizes[head] == 0 {
            out_pos[head] = written;
            cursor += 1;
            continue;
        }
        // Start a run at this span and extend it while the next *distinct* span is heap-adjacent.
        // Duplicate spans (identical offset+size) reuse the position already assigned for the run.
        let run_out_base = written;
        let run_heap_start = offsets[head];
        let mut run_heap_end = run_heap_start;
        let mut elem_out = written;
        while cursor < count {
            let elem = order[cursor];
            if sizes[elem] == 0 {
                break;
            }
            if offsets[elem] == run_heap_end {
                // A new distinct span that continues the run.
                out_pos[elem] = elem_out;
                elem_out += ulens[elem];
                run_heap_end += sizes[elem];
                cursor += 1;
            } else if offsets[elem] < run_heap_end {
                // A duplicate of a span already decoded in this run: reuse its position. Duplicates
                // are contiguous in the sorted order, so the previous entry shares this span.
                out_pos[elem] = out_pos[order[cursor - 1]];
                cursor += 1;
            } else {
                break;
            }
        }
        // One decode for the whole run, straight into the output at `run_out_base`.
        decompressor.decompress_into(
            &heap[run_heap_start..run_heap_end],
            &mut spare[run_out_base..],
        );
        written = elem_out;
    }
    unsafe { out.set_len(written) };
    let bytes = out.freeze();

    // Build views in element order, each pointing at its decoded output position.
    let mut views = BufferMut::<BinaryView>::with_capacity(count);
    for (i, &ulen) in ulens.iter().enumerate() {
        let pos = out_pos[i];
        #[expect(clippy::cast_possible_truncation)]
        let view = BinaryView::make_view(&bytes[pos..pos + ulen], 0, pos as u32);
        views.push(view);
    }

    (vec![bytes], views.freeze())
}

/// Decompress each element's code slice directly into its place in the output (no compaction).
fn decompress_per_element(
    decompressor: &Decompressor<'_>,
    heap: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    ulens: &[usize],
    total_size: usize,
) -> ByteBufferMut {
    let mut out = ByteBufferMut::with_capacity(total_size + 7);
    {
        let spare = out.spare_capacity_mut();
        let mut uoff = 0;
        for ((&offset, &size), &ulen) in offsets.iter().zip(sizes).zip(ulens) {
            if size > 0 {
                decompressor.decompress_into(&heap[offset..offset + size], &mut spare[uoff..]);
            }
            uoff += ulen;
        }
    }
    unsafe { out.set_len(total_size) };
    out
}
