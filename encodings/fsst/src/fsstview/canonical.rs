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
//!
//! ## Export heuristic: "export all in place" vs "compact codes"
//!
//! `GatherBulk` always copies the live codes contiguous before decoding. But after a `filter`, a
//! sorted-index `take`, or a `slice`, the survivors' offsets stay **monotonic** — so we can skip
//! the gather entirely and decode each contiguous heap run *directly into* the (element-ordered)
//! output: [`RunDecode`][FsstViewCompaction::RunDecode]. Unlike `RunCoalesce`, the output is in
//! element order, so the view-build stays sequential. The cost is one decode call per run, so it
//! wins while survivors form few runs (clustered / range selections) and loses once they fragment
//! into many tiny runs (a uniform-random filter), where one bulk decode (`GatherBulk`) is cheaper.
//!
//! `Auto` therefore decides between *exporting all in place* and *compacting codes then exporting*
//! by **run count**: `RunDecode` when `runs <= len / RUN_DECODE_MAX_RUN_FRACTION` (and the layout
//! is monotonic), else `GatherBulk`. The `db_*`/`canon_only` benches calibrate this: on
//! `many_short` it's RunDecode ~313 µs (clustered) / ~345 µs (range) vs GatherBulk ~333 / ~370 µs,
//! and GatherBulk ~561 µs vs RunDecode ~657 µs on uniform-random. Crucially this lives entirely in
//! the export — the conversion and the metadata-only `filter`/`take` stay separate so a *chain* of
//! them still composes; only the final canonicalize compacts (or not).

use std::sync::Arc;

use fsst::Decompressor;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::buffer::BufferHandle;
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
    /// "Export all in place": when survivors are in heap order (offsets monotonically increasing,
    /// as after any `filter`, a sorted-index `take`, or a `slice`), decode each maximal contiguous
    /// heap run *directly* into the element-ordered output, with **no gather copy**. The output is
    /// element-ordered, so the view-build stays sequential (unlike `RunCoalesce`). Cost is one
    /// decode call per run; it beats `GatherBulk` when survivors form few runs (clustered/range
    /// selections), and degrades toward per-element decode when survivors are scattered (a
    /// uniform-random filter), which is when `GatherBulk`'s single bulk decode wins instead.
    ///
    /// Requires monotonic offsets; falls back to `GatherBulk` otherwise (e.g. a shuffle take).
    RunDecode,
}

pub(super) fn canonicalize_fsstview(
    array: ArrayView<'_, FSSTView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    canonicalize_fsstview_with(array, FsstViewCompaction::Auto, ctx)
}

/// Byte accounting for an [`FSSTView`], in **both compressed (code) space and uncompressed
/// (decoded) space**, for reasoning about gather/coalesce trade-offs and dead-byte waste.
///
/// All figures are in bytes. The "span" figures describe what a *gap-merged* decode (decoding each
/// run's full heap extent, dead bytes included) would touch; the difference from the live figures
/// is the waste such a strategy would carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FsstViewByteStats {
    /// Number of (logical) elements in the view.
    pub elements: usize,
    /// Distinct heap runs the live elements form (maximal heap-adjacent groups of distinct spans).
    pub runs: usize,
    /// Distinct (deduplicated) live code spans referenced by the view.
    pub distinct_spans: usize,
    /// Compressed bytes the live distinct spans occupy (what `GatherBulk` copies / decodes).
    pub live_compressed: usize,
    /// Compressed bytes spanned by the runs *including* dead gaps between survivors (what a
    /// gap-merged decode would feed the decoder). `span_compressed - live_compressed` is the
    /// compressed waste of merging across gaps.
    pub span_compressed: usize,
    /// Uncompressed bytes the live elements decode to (the canonical output size; deduped spans
    /// counted once).
    pub live_uncompressed: usize,
    /// Total uncompressed output size with duplicates expanded (the `VarBinView`'s logical size).
    pub logical_uncompressed: usize,
    /// Total compressed heap size backing the view (the original, shared code buffer).
    pub heap_compressed: usize,
}

impl FsstViewByteStats {
    /// Fraction of the spanned compressed bytes that are dead (would be wasted by a gap-merged
    /// decode). `0.0` means the live spans are perfectly contiguous within each run.
    pub fn compressed_waste_ratio(&self) -> f64 {
        if self.span_compressed == 0 {
            0.0
        } else {
            (self.span_compressed - self.live_compressed) as f64 / self.span_compressed as f64
        }
    }
}

/// Compute [`FsstViewByteStats`] for a view (diagnostics; not on the hot path).
pub fn fsstview_byte_stats(
    array: ArrayView<'_, FSSTView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FsstViewByteStats> {
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

    let elements = offsets.len();
    let logical_uncompressed: usize = ulens.iter().sum();
    let heap_compressed = array.codes_bytes().len();

    // Walk distinct spans in heap order, accumulating live/run/span figures.
    let mut order: Vec<usize> = (0..elements).filter(|&i| sizes[i] > 0).collect();
    order.sort_unstable_by_key(|&i| (offsets[i], sizes[i]));

    let mut runs = 0usize;
    let mut distinct_spans = 0usize;
    let mut live_compressed = 0usize;
    let mut live_uncompressed = 0usize;
    let mut span_compressed = 0usize;
    let mut run_end: Option<usize> = None;
    let mut run_start = 0usize;
    let mut prev_span: Option<(usize, usize)> = None;
    for &i in &order {
        let span = (offsets[i], sizes[i]);
        let is_dup = prev_span == Some(span);
        prev_span = Some(span);
        if is_dup {
            continue; // duplicate of the previous distinct span
        }
        distinct_spans += 1;
        live_compressed += sizes[i];
        live_uncompressed += ulens[i];
        match run_end {
            Some(end) if offsets[i] == end => {
                run_end = Some(end + sizes[i]);
            }
            Some(end) => {
                // Close the previous run, open a new one.
                span_compressed += end - run_start;
                runs += 1;
                run_start = offsets[i];
                run_end = Some(offsets[i] + sizes[i]);
            }
            None => {
                run_start = offsets[i];
                run_end = Some(offsets[i] + sizes[i]);
            }
        }
    }
    if let Some(end) = run_end {
        span_compressed += end - run_start;
        runs += 1;
    }

    Ok(FsstViewByteStats {
        elements,
        runs,
        distinct_spans,
        live_compressed,
        span_compressed,
        live_uncompressed,
        logical_uncompressed,
        heap_compressed,
    })
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

    // Analyse the survivor layout once: a single contiguous run (Direct), monotonic-but-gapped
    // (RunDecode candidate), or out of heap order (must gather).
    let layout = analyze_layout(&offsets, &sizes);
    let chosen = match strategy {
        // The export heuristic. With monotonic offsets we can "export all in place" by decoding
        // each contiguous run with no gather copy; this wins while the runs are few. Once survivors
        // fragment into many tiny runs (a uniform-random filter), the per-run decode-tail overhead
        // dominates and compacting the codes into one bulk decode (`GatherBulk`) wins instead.
        // Non-monotonic layouts (a shuffle take) can't run-decode, so they always gather.
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

    if chosen == FsstViewCompaction::RunDecode {
        let uncompressed =
            decompress_run_decode(&decompressor, heap, &offsets, &sizes, &ulens, total_size);
        let (buffers, views) = match_each_integer_ptype!(ulen_prim.ptype(), |P| {
            build_views(0, MAX_BUFFER_LEN, uncompressed, ulen_prim.as_slice::<P>())
        });
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

/// Canonicalize an [`FSSTView`] to a [`VarBinArray`] (offsets + contiguous bytes) instead of a
/// [`VarBinViewArray`].
///
/// Shares the decode path with [`canonicalize_fsstview_with`]: the strategies that produce an
/// element-ordered output (`Direct`/`GatherBulk`/`PerElement`) are reused as-is; the only
/// difference is the finisher, which builds `len + 1` cumulative offsets from the uncompressed
/// lengths rather than per-element views. `RunCoalesce` is not applicable (its output is heap-
/// ordered, not element-ordered) and is treated as `GatherBulk`.
///
/// Exposed for benchmarking the export target (VarBin vs VarBinView). `Auto` resolves to `Direct`
/// when contiguous, else `GatherBulk`.
pub fn canonicalize_fsstview_to_varbin(
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
    let uncompressed = match strategy {
        FsstViewCompaction::PerElement => {
            decompress_per_element(&decompressor, heap, &offsets, &sizes, &ulens, total_size)
        }
        // Direct (or Auto) on a contiguous layout decodes the live range in place, no gather.
        FsstViewCompaction::Direct | FsstViewCompaction::Auto if contiguous => {
            let start = offsets.first().copied().unwrap_or(0);
            decompress_direct(&decompressor, heap, start, live, total_size)
        }
        // Everything else uses the element-ordered (coalesced) gather + one bulk decode.
        _ => decompress_gather(&decompressor, heap, &offsets, &sizes, live, total_size),
    };

    // Build `len + 1` cumulative offsets from the uncompressed lengths.
    let mut varbin_offsets = BufferMut::<i64>::with_capacity(ulens.len() + 1);
    let mut acc = 0i64;
    varbin_offsets.push(acc);
    for &ulen in &ulens {
        acc += ulen as i64;
        varbin_offsets.push(acc);
    }

    let bytes = BufferHandle::new_host(uncompressed.freeze());
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

/// `Auto` prefers `RunDecode` (export all in place) over `GatherBulk` (compact codes) while the
/// number of contiguous runs is at most `len / RUN_DECODE_MAX_RUN_FRACTION` — i.e. while survivors
/// average more than this many elements per run. Calibrated by the `db_*` benchmarks: clustered
/// and range selections sit well under this, uniform-random filters well over it.
const RUN_DECODE_MAX_RUN_FRACTION: usize = 4;

/// The survivor layout in the heap, used to pick an export strategy.
enum Layout {
    /// Survivors are one contiguous in-order run (untouched / sliced view) — `Direct`.
    Contiguous,
    /// Offsets are strictly increasing but gapped: survivors form `runs` contiguous blocks.
    /// Eligible for `RunDecode` (decode each run in place, no gather).
    Monotonic { runs: usize },
    /// Offsets are out of heap order (e.g. a shuffle take) — must gather.
    Scattered,
}

/// Classify the survivor layout in a single O(n) pass: are offsets monotonic, and how many
/// maximal contiguous runs do the (non-empty) survivors form?
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
    if !gapped {
        Layout::Contiguous
    } else {
        Layout::Monotonic { runs }
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
