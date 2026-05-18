// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pruning unreferenced elements from a [`ListViewArray`].
//!
//! `take` and `filter` on a [`ListViewArray`] only rewrite the `offsets`/`sizes`/`validity`
//! children — the `elements` buffer is preserved as-is, regardless of whether the surviving
//! views still cover all of it. Consumers that materialise the elements (Arrow conversion, the
//! DuckDB exporter, …) then pay to decompress every element even when only a small fraction is
//! referenced.
//!
//! [`maybe_prune_unreferenced_elements`] estimates how much of the elements buffer is reachable
//! and, if a sizable fraction is unreferenced, delegates to
//! [`rebuild`](super::ListViewArray::rebuild) with
//! [`ListViewRebuildMode::MakeZeroCopyToList`]. The rebuild path uses `take` to materialise only
//! the referenced positions, which means compressed elements stay compressed for the discarded
//! ones.
//!
//! ### How the reachable estimate is obtained
//!
//! The decision walk consults [`ListViewArrayExt::reachable_elements_bound`] first. The bound is
//! maintained by `rebuild`, `take`, `slice`, and `prune_unreferenced_elements` as their byproduct
//! — each of these ops already touches `sizes`, so summing the kept sizes adds only the
//! per-element work. The result is an upper bound on the reachable count (overlapping or
//! duplicate views in `take` can shrink the true count further).
//!
//! If the bound is `None` (e.g. the array was constructed directly without going through one of
//! the ops above), we fall back to materialising `sizes` and computing the sum here.

use num_traits::AsPrimitive;
use vortex_error::VortexResult;

use super::ListViewArray;
use super::ListViewArrayExt;
use super::ListViewRebuildMode;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::match_each_integer_ptype;

/// Minimum elements buffer length before pruning is even considered, when elements are in a
/// canonical (host) encoding.
///
/// For canonical elements the downstream export is close to a `memcpy` — the rebuild's fixed
/// cost only pays back on sizable buffers. Calibrated conservatively against an export of
/// `Vector::with_capacity(elements.len())`.
pub const PRUNE_LISTVIEW_MIN_ELEMENTS_CANONICAL: usize = 64 * 1024;

/// Minimum elements buffer length before pruning is even considered, when elements are
/// **non-canonical** (e.g. `Dict`, `Fsst`, `RunEnd`, …).
///
/// Compressed elements pay per-position decompression at export time, but the rebuild itself
/// has fixed overhead (allocations, the child encoding's own take + post-take pruning). For
/// very small compressed buffers the linear export cost is comparable to or smaller than the
/// rebuild's fixed cost, so we still skip there.
pub const PRUNE_LISTVIEW_MIN_ELEMENTS_COMPRESSED: usize = 16 * 1024;

/// Maximum fraction of `elements.len()` that may be reachable via `sizes` and still be worth
/// pruning, for canonical elements.
///
/// For canonical export the per-element cost is roughly a `memcpy`, so the rebuild only wins
/// when the unreferenced portion is overwhelming.
pub const PRUNE_LISTVIEW_MAX_REFERENCED_RATIO_CANONICAL: f64 = 0.02;

/// Maximum fraction of `elements.len()` that may be reachable via `sizes` and still be worth
/// pruning, for compressed elements.
///
/// Compressed exports do per-position decompression, but the rebuild itself runs the child
/// encoding's take + post-take prune, which has its own per-element cost. Empirically the
/// break-even ratio sits around 30–40% for dict-encoded varbin; 10% leaves comfortable slack so
/// the rebuild reliably pays back without trimming the genuine wins (sparse compressed sources
/// at <5% reachable still trip easily).
pub const PRUNE_LISTVIEW_MAX_REFERENCED_RATIO_COMPRESSED: f64 = 0.10;

/// Inspect `array` and, if a sizable fraction of the elements buffer is unreferenced, return a
/// rebuilt [`ListViewArray`] whose elements contain only the referenced positions.
///
/// Decisions are made in three stages:
///
/// 1. **Encoding-aware thresholds.** Pick `MIN_ELEMENTS` / `MAX_REFERENCED_RATIO` based on
///    whether `elements` is in a canonical encoding (cheap to export) or a compressed one
///    (per-position decompression dominates). Compressed children justify a much more
///    aggressive prune.
/// 2. **Cheap rejections.** Skip when the elements buffer is below the chosen `MIN_ELEMENTS`,
///    or when the array is already `is_zero_copy_to_list` (offsets are sequential with no gaps
///    or overlaps), so a rebuild would not change the elements buffer.
/// 3. **Sum sniff.** Read the propagated `reachable_elements_bound` (O(1)) or, if absent,
///    canonicalise `sizes` and compute the total — bounded by `array.len()`, typically
///    DuckDB's vector size. `sum(sizes)` is a strict upper bound on the survivor count, so a
///    small ratio is a sufficient signal to commit.
///
/// Returns `Ok(None)` to mean "no change worth making"; callers should keep the original array.
pub fn maybe_prune_unreferenced_elements(
    array: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ListViewArray>> {
    let elements_len = array.elements().len();
    let canonical_elements = array.elements().is_canonical();
    let (min_elements, max_ratio_inv): (usize, u64) = if canonical_elements {
        // 1.0 / 0.02 = 50
        (PRUNE_LISTVIEW_MIN_ELEMENTS_CANONICAL, 50)
    } else {
        // 1.0 / 0.10 = 10
        (PRUNE_LISTVIEW_MIN_ELEMENTS_COMPRESSED, 10)
    };
    if elements_len < min_elements {
        return Ok(None);
    }
    // Already zero-copy: sequential offsets, no gaps, no overlaps. The rebuild path would just
    // return a clone, so don't pay for the size scan.
    if array.is_zero_copy_to_list() {
        return Ok(None);
    }

    // Bound on the reachable element count. First consult the metadata hint that
    // `take`/`slice`/`rebuild` maintain — if present it's O(1) here. Otherwise canonicalise
    // `sizes` and sum, which is bounded by `num_lists` (usually <= DuckDB's vector size).
    let total_referenced: u64 = match array.reachable_elements_bound() {
        Some(b) => b,
        None => {
            let sizes = array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(sizes.ptype(), |S| {
                sizes
                    .as_slice::<S>()
                    .iter()
                    .map(|s| AsPrimitive::<u64>::as_(*s))
                    .sum()
            })
        }
    };

    let elements_len_u64 = elements_len as u64;
    // Strict upper bound on the rebuild output size. If even this exceeds the threshold there's
    // no point continuing. `total_referenced * (1/max_ratio) >= elements_len` iff
    // `referenced/elements >= max_ratio`.
    if elements_len_u64 == 0 || total_referenced * max_ratio_inv >= elements_len_u64 {
        return Ok(None);
    }

    // Acceptable savings — perform the rebuild. `MakeZeroCopyToList` uses `take` for small
    // average list sizes, which means compressed `elements` only get decompressed for the
    // positions we actually keep.
    let rebuilt = array.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?;
    Ok(Some(rebuilt))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn returns_none_for_small_elements() -> VortexResult<()> {
        let lv = ListViewArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 4].into_array(),
            buffer![2u32, 2, 2].into_array(),
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_prune_unreferenced_elements(&lv, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn returns_none_when_dense() -> VortexResult<()> {
        // 4096 elements with 2048 lists each of size 2 fully covers the elements (ratio = 1.0).
        let elements = PrimitiveArray::from_iter(0i64..4096).into_array();
        let offsets: Vec<u32> = (0..2048).map(|i| (i * 2) as u32).collect();
        let sizes = vec![2u32; 2048];
        let lv = ListViewArray::new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            PrimitiveArray::from_iter(sizes).into_array(),
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(maybe_prune_unreferenced_elements(&lv, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn prunes_when_elements_are_sparse() -> VortexResult<()> {
        // 262_144 elements, 2048 lists each of size 1: 2048/262144 ≈ 0.8% referenced.
        let element_count = 262_144usize;
        let elements = PrimitiveArray::from_iter(0i64..(element_count as i64)).into_array();
        // Spread the offsets across the buffer so we exercise the take path.
        let offsets: Vec<u32> = (0..2048).map(|i| (i * 128) as u32).collect();
        let sizes = vec![1u32; 2048];
        let lv = ListViewArray::new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            PrimitiveArray::from_iter(sizes).into_array(),
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let pruned = maybe_prune_unreferenced_elements(&lv, &mut ctx)?
            .expect("expected prune for sparse listview");
        assert!(pruned.elements().len() <= 2048);
        assert!(pruned.is_zero_copy_to_list());
        Ok(())
    }

    #[test]
    fn skips_already_zero_copy() -> VortexResult<()> {
        // Make a zero-copy ListView with leading garbage (should be filtered by trim, not prune).
        let element_count = 10_000usize;
        let elements = PrimitiveArray::from_iter(0i64..(element_count as i64)).into_array();
        let offsets: Vec<u32> = (0..2048).map(|i| i as u32).collect();
        let sizes = vec![1u32; 2048];
        let lv = unsafe {
            ListViewArray::new(
                elements,
                PrimitiveArray::from_iter(offsets).into_array(),
                PrimitiveArray::from_iter(sizes).into_array(),
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Even though referenced is only 20%, we skip because it's already zero-copy. Trimming
        // is the right tool here, but that's a separate optimization handled at the rebuild
        // layer.
        assert!(maybe_prune_unreferenced_elements(&lv, &mut ctx)?.is_none());
        Ok(())
    }
}
