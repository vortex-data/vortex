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
//!
//! The compaction question, concretely. The `fsst_view_compute` benchmark (two ~2 MiB inputs,
//! ~12-byte and ~256-byte strings) shows **`GatherBulk` beats `PerElement` across the whole
//! tested range, for both short and long strings**. The reason: FSST's decoder has a fast 8-wide
//! body and a slow byte-by-byte tail. `PerElement` pays that tail *once per element* (N tails),
//! while `GatherBulk` decodes the whole heap in one call and pays the tail *once*. That saving
//! dominates the cost of the gather memcpy even at 256-byte strings (with ~8 K elements). For
//! example `take few_long/shuffle` canonicalizes in ~459 µs with `GatherBulk` vs ~623 µs with
//! `PerElement`.
//!
//! `PerElement` only wins in the opposite extreme — *very few, very long* strings — where N is
//! tiny (few tails saved) but the gather memcpy of the entire live heap is large. That regime is
//! outside what real string columns hit, so [`FsstViewCompaction::Auto`] never picks it: it uses
//! `Direct` when the live codes are still contiguous (untouched/sliced view) and `GatherBulk`
//! otherwise. `PerElement` is kept selectable so the trade-off stays measurable.

use std::sync::Arc;

use fsst::Decompressor;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
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
    /// `GatherBulk`. Never picks `PerElement` (see module docs).
    Auto,
    /// Bulk-decompress the contiguous live range with no copy. Falls back to `GatherBulk` if the
    /// view's codes are not contiguous and in order.
    Direct,
    /// Compact the scattered live codes into a contiguous buffer, then a single bulk decompress.
    GatherBulk,
    /// Decompress each element's code slice directly into place, without compacting.
    PerElement,
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
        // `GatherBulk` beats `PerElement` across the whole practical range (see module docs), so
        // `Auto` never selects `PerElement`.
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

    let uncompressed = match chosen {
        FsstViewCompaction::Direct => {
            let start = offsets.first().copied().unwrap_or(0);
            decompress_direct(&decompressor, heap, start, live, total_size)
        }
        FsstViewCompaction::GatherBulk => {
            decompress_gather(&decompressor, heap, &offsets, &sizes, live, total_size)
        }
        // `Auto` is always resolved to a concrete strategy above.
        FsstViewCompaction::PerElement | FsstViewCompaction::Auto => {
            decompress_per_element(&decompressor, heap, &offsets, &sizes, &ulens, total_size)
        }
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
