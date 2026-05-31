// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalization of [`FSSTView`] into a [`VarBinViewArray`] (or [`VarBinArray`]).
//!
//! After metadata-only `filter`/`take`/`slice`, an [`FSSTView`]'s byte heap is the *original* heap
//! and the live codes are scattered (gaps after a filter, reordering/duplication after a take). To
//! canonicalize we must decode the survivors into one element-ordered buffer. [`FsstViewCompaction`]
//! captures how:
//!
//! - [`Direct`][FsstViewCompaction::Direct]: the live codes are still one contiguous in-order run
//!   (an untouched or sliced view). Decode that single range in one call, no copy.
//! - [`RunDecode`][FsstViewCompaction::RunDecode] ("export all in place"): the offsets are still
//!   monotonic (after any `filter`, sorted-index `take`, or `slice`) but gapped. Decode each
//!   maximal contiguous heap run *directly* into the element-ordered output, with **no gather
//!   copy** — one decode call per run. Wins while survivors form few runs (clustered / range
//!   selections).
//! - [`GatherBulk`][FsstViewCompaction::GatherBulk] ("compact codes"): for scattered survivors (a
//!   shuffle take) or heavily fragmented ones (a uniform-random filter), compact the live codes
//!   into one contiguous buffer, then a single bulk decode. The one bulk call amortizes FSST's slow
//!   decode tail across all elements, which beats run-decode once the runs get small.
//!
//! [`FsstViewCompaction::Auto`] picks `Direct` when contiguous, `RunDecode` when the offsets are
//! monotonic and the survivors form few runs (`runs <= len / RUN_DECODE_MAX_RUN_FRACTION`), and
//! `GatherBulk` otherwise. The choice lives entirely in the export: the conversion and the
//! metadata-only `filter`/`take` stay separate so a *chain* of them composes; only the final
//! canonicalize compacts (or not).

use std::sync::Arc;

use fsst::Decompressor;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::buffer::BufferHandle;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use super::array::FSSTView;
use super::array::FSSTViewArrayExt;
use super::array::FSSTViewArraySlotsExt;

/// Strategy for materializing the decompressed bytes when canonicalizing an [`FSSTView`].
///
/// See the [module docs][self] for the full trade-off analysis. Every strategy produces an
/// element-ordered decoded buffer; they differ only in how the survivor codes are fed to the
/// decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsstViewCompaction {
    /// Pick automatically: `Direct` when contiguous, `RunDecode` when monotonic with few runs,
    /// else `GatherBulk`.
    Auto,
    /// Bulk-decode the single contiguous live range, no copy. Falls back to `GatherBulk` if the
    /// codes are not contiguous and in order.
    Direct,
    /// Compact the scattered live codes into a contiguous buffer, then a single bulk decode.
    GatherBulk,
    /// Decode each contiguous heap run directly into the element-ordered output, no gather copy.
    /// Requires monotonic offsets; falls back to `GatherBulk` otherwise (e.g. a shuffle take).
    RunDecode,
}

/// `Auto` prefers `RunDecode` (export all in place) over `GatherBulk` (compact codes) while the
/// number of contiguous runs is at most `len / RUN_DECODE_MAX_RUN_FRACTION` — i.e. while survivors
/// average more than this many elements per run. Calibrated by the `fsst_view_compute` benches:
/// clustered and range selections sit well under this, uniform-random filters well over it.
const RUN_DECODE_MAX_RUN_FRACTION: usize = 4;

pub(super) fn canonicalize_fsstview(
    array: ArrayView<'_, FSSTView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    canonicalize_fsstview_with(array, FsstViewCompaction::Auto, ctx)
}

/// Canonicalize an [`FSSTView`] to a [`VarBinViewArray`] using an explicit compaction strategy.
///
/// Exposed (rather than only the dispatch-driven [`canonicalize_fsstview`]) so callers and
/// benchmarks can force a strategy. Production code goes through [`FsstViewCompaction::Auto`].
pub fn canonicalize_fsstview_with(
    array: ArrayView<'_, FSSTView>,
    strategy: FsstViewCompaction,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decoded = decode_element_ordered(array, strategy, ctx)?;
    let (buffers, views) = match_each_integer_ptype!(decoded.ulen_prim.ptype(), |P| {
        build_views(
            0,
            MAX_BUFFER_LEN,
            decoded.uncompressed,
            decoded.ulen_prim.as_slice::<P>(),
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

/// Canonicalize an [`FSSTView`] to a [`VarBinArray`] (offsets + contiguous bytes) instead of a
/// [`VarBinViewArray`].
///
/// Shares the element-ordered decode path with [`canonicalize_fsstview_with`]; the only difference
/// is the finisher, which builds `len + 1` cumulative offsets from the uncompressed lengths rather
/// than per-element views. Cheaper than a `VarBinViewArray` when the consumer wants offsets+bytes
/// (no per-element 16-byte view construction).
pub fn canonicalize_fsstview_to_varbin(
    array: ArrayView<'_, FSSTView>,
    strategy: FsstViewCompaction,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decoded = decode_element_ordered(array, strategy, ctx)?;

    let varbin_offsets = cumulative_offsets(&decoded.ulen_prim);
    let bytes = BufferHandle::new_host(decoded.uncompressed.freeze());
    // SAFETY: offsets are monotonic and end at the byte length; bytes are valid binary/UTF-8.
    Ok(unsafe {
        VarBinArray::new_unchecked_from_handle(
            varbin_offsets.into_array(),
            bytes,
            array.dtype().clone(),
            array.fsstview_validity(),
        )
        .into_array()
    })
}

/// The element-ordered decoded bytes plus the uncompressed-lengths array the finishers need.
struct Decoded {
    uncompressed: ByteBufferMut,
    ulen_prim: PrimitiveArray,
}

/// Decode an [`FSSTView`]'s survivors into one element-ordered buffer using the chosen (or `Auto`)
/// strategy. Shared by the `VarBinView` and `VarBin` finishers.
fn decode_element_ordered(
    array: ArrayView<'_, FSSTView>,
    strategy: FsstViewCompaction,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Decoded> {
    let offsets = load_usize(array.codes_offsets(), ctx)?;
    // Derive each survivor's size in place from its end offset (`codes_ends[i] - codes_offsets[i]`),
    // reusing the widened `ends` buffer as `sizes` so we don't allocate a third index array.
    // Downstream layout analysis and decode work on `sizes` exactly as before.
    let mut sizes = load_usize(array.codes_ends(), ctx)?;
    for (size, &offset) in sizes.iter_mut().zip(&offsets) {
        *size -= offset;
    }

    let ulen_prim = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    // `total_size` is needed by every path; sum it from the typed slice. The widened
    // `ulens: Vec<usize>` is only needed by `RunDecode`, so defer it.
    #[expect(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
        ulen_prim.as_slice::<P>().iter().map(|x| *x as usize).sum()
    });

    let heap_buffer = array.codes_bytes();
    let heap = heap_buffer.as_slice();
    let decompressor = array.decompressor();

    let layout = analyze_layout(&offsets, &sizes);
    let chosen = match strategy {
        FsstViewCompaction::Auto => match layout {
            Layout::Contiguous => FsstViewCompaction::Direct,
            Layout::Monotonic { runs } if runs <= offsets.len() / RUN_DECODE_MAX_RUN_FRACTION => {
                FsstViewCompaction::RunDecode
            }
            _ => FsstViewCompaction::GatherBulk,
        },
        // `Direct`/`RunDecode` require a (contiguous / monotonic) layout; fall back to gather.
        FsstViewCompaction::Direct if !matches!(layout, Layout::Contiguous) => {
            FsstViewCompaction::GatherBulk
        }
        FsstViewCompaction::RunDecode if matches!(layout, Layout::Scattered) => {
            FsstViewCompaction::GatherBulk
        }
        other => other,
    };

    let uncompressed = match chosen {
        FsstViewCompaction::Direct => {
            let start = offsets.first().copied().unwrap_or(0);
            // `live` (total compressed bytes) is only needed by the bulk-decode paths, not by
            // `RunDecode`, so it is summed here rather than unconditionally up front.
            let live: usize = sizes.iter().sum();
            decompress_direct(&decompressor, heap, start, live, total_size)
        }
        FsstViewCompaction::RunDecode => {
            let ulens = widen_ulens(&ulen_prim);
            decompress_run_decode(&decompressor, heap, &offsets, &sizes, &ulens, total_size)
        }
        // `Auto` is resolved above; `GatherBulk` is the catch-all.
        _ => {
            let live: usize = sizes.iter().sum();
            decompress_gather(&decompressor, heap, &offsets, &sizes, live, total_size)
        }
    };

    Ok(Decoded {
        uncompressed,
        ulen_prim,
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

/// Build `len + 1` cumulative offsets over the uncompressed lengths (the `VarBin` offsets array),
/// directly from the typed slice. `push_unchecked` (capacity reserved) keeps this vectorized.
fn cumulative_offsets(ulen_prim: &PrimitiveArray) -> ArrayRef {
    let len = ulen_prim.len();
    let mut offsets = BufferMut::<i64>::with_capacity(len + 1);
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let _: () = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
        let mut acc: usize = 0;
        // SAFETY: `len + 1` slots reserved; we push exactly that many.
        unsafe { offsets.push_unchecked(0) };
        for &ulen in ulen_prim.as_slice::<P>() {
            acc += ulen as usize;
            unsafe { offsets.push_unchecked(acc as i64) };
        }
    });
    offsets.into_array()
}

/// Widen an already-executed uncompressed-lengths primitive array into `Vec<usize>`. Only
/// `RunDecode` needs this; `Direct`/`GatherBulk` work without it.
fn widen_ulens(ulen_prim: &PrimitiveArray) -> Vec<usize> {
    #[expect(clippy::cast_possible_truncation)]
    let out: Vec<usize> = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
        ulen_prim
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .collect()
    });
    out
}

/// The survivor layout in the heap, used to pick an export strategy.
enum Layout {
    /// Survivors are one contiguous in-order run (untouched / sliced view) — `Direct`.
    Contiguous,
    /// Offsets are strictly increasing but gapped: survivors form `runs` contiguous blocks.
    /// Eligible for `RunDecode`.
    Monotonic { runs: usize },
    /// Offsets are out of heap order (e.g. a shuffle take) — must gather.
    Scattered,
}

/// Classify the survivor layout in a single O(n) pass: are offsets monotonic, and how many maximal
/// contiguous runs do the (non-empty) survivors form?
fn analyze_layout(offsets: &[usize], sizes: &[usize]) -> Layout {
    let mut runs = 0usize;
    let mut gapped = false;
    let mut prev_end: Option<usize> = None;
    for (&offset, &size) in offsets.iter().zip(sizes) {
        if size == 0 {
            continue; // empty/null elements don't affect run structure
        }
        match prev_end {
            None => runs = 1,
            Some(end) if offset == end => {} // continues the current run
            Some(end) if offset > end => {
                runs += 1;
                gapped = true;
            }
            Some(_) => return Layout::Scattered, // offset < end: out of order
        }
        prev_end = Some(offset + size);
    }
    if gapped {
        Layout::Monotonic { runs }
    } else {
        Layout::Contiguous
    }
}

/// "Export all in place": decode each maximal contiguous heap run directly into the element-ordered
/// output, with no gather copy. Requires monotonic offsets (the caller guarantees this).
fn decompress_run_decode(
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
        // Walk elements in order, batching heap-adjacent survivors into one decode call. `out_pos`
        // tracks where the current run's decoded bytes begin in the (element-ordered) output.
        let mut out_pos = 0usize;
        let mut i = 0usize;
        while i < offsets.len() {
            if sizes[i] == 0 {
                i += 1;
                continue;
            }
            let run_heap_start = offsets[i];
            let mut run_heap_end = run_heap_start;
            let mut run_uncompressed = 0usize;
            let mut j = i;
            while j < offsets.len() {
                if sizes[j] == 0 {
                    j += 1;
                    continue;
                }
                if offsets[j] != run_heap_end {
                    break;
                }
                run_heap_end += sizes[j];
                run_uncompressed += ulens[j];
                j += 1;
            }
            decompressor
                .decompress_into(&heap[run_heap_start..run_heap_end], &mut spare[out_pos..]);
            out_pos += run_uncompressed;
            i = j;
        }
    }
    unsafe { out.set_len(total_size) };
    out
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
///
/// The gather coalesces consecutive heap-adjacent spans into a single `extend_from_slice`: for an
/// order-preserving `filter`, surviving neighbours are still contiguous in the heap, so a run of
/// `k` survivors is copied in one memcpy instead of `k`. This collapses the per-span copy overhead
/// (which dominates for short codes) to per-run, while a shuffle (no adjacency) is unaffected.
fn decompress_gather(
    decompressor: &Decompressor<'_>,
    heap: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    live: usize,
    total_size: usize,
) -> ByteBufferMut {
    let mut compressed = ByteBufferMut::with_capacity(live);
    // Accumulate a contiguous `[run_start, run_end)` heap range and flush it as one copy.
    let mut run_start = 0usize;
    let mut run_end = 0usize;
    for (&offset, &size) in offsets.iter().zip(sizes) {
        if size == 0 {
            continue;
        }
        if offset == run_end && run_end != run_start {
            run_end += size; // extend the current run (heap-adjacent)
        } else {
            if run_end != run_start {
                compressed.extend_from_slice(&heap[run_start..run_end]);
            }
            run_start = offset;
            run_end = offset + size;
        }
    }
    if run_end != run_start {
        compressed.extend_from_slice(&heap[run_start..run_end]);
    }
    let mut out = ByteBufferMut::with_capacity(total_size + 7);
    let written = decompressor.decompress_into(compressed.as_slice(), out.spare_capacity_mut());
    unsafe { out.set_len(written) };
    out
}
