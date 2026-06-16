// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared run-end "ends indexing" logic reused by both the `RunEnd` and `RunEndBool` encodings.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::scalar::PValue;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Shared run-end index math over a strictly-increasing unsigned `ends` child.
pub trait RunEndIndex {
    /// The strictly-increasing run-end positions.
    fn ends(&self) -> &ArrayRef;

    /// The logical offset into the first run.
    fn offset(&self) -> usize;

    /// Find the physical run index containing logical `index`.
    fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        find_physical_index(self.ends(), self.offset(), index)
    }
}

/// Find the physical run index containing logical `index` for the given `ends` child and `offset`.
///
/// This is the free-function form of [`RunEndIndex::find_physical_index`], usable where the orphan
/// rule prevents implementing [`RunEndIndex`] for a foreign array type.
pub fn find_physical_index(ends: &ArrayRef, offset: usize, index: usize) -> VortexResult<usize> {
    Ok(ends
        .as_primitive_typed()
        .search_sorted(&PValue::from(index + offset), SearchSortedSide::Right)?
        .to_ends_index(ends.len()))
}

/// Find the physical offset for an index that would be an end of the slice, i.e., one past the last
/// element.
///
/// If the index exists in the array we take that position (as we are searching from the right);
/// otherwise we take the next one.
pub fn find_slice_end_index(ends: &ArrayRef, index: usize) -> VortexResult<usize> {
    let result = ends
        .as_primitive_typed()
        .search_sorted(&PValue::from(index), SearchSortedSide::Right)?;
    Ok(match result {
        SearchResult::Found(i) => i,
        SearchResult::NotFound(i) => {
            if i == ends.len() {
                i
            } else {
                i + 1
            }
        }
    })
}

/// Validate the common invariants of a run-end `ends` child for the given `offset` and `length`.
///
/// This checks that the ends are unsigned integers, that an empty ends child implies a zero offset,
/// and (for host-resident, non-empty, non-zero-length arrays) that the offset and last run end are
/// within bounds. In debug builds it additionally asserts that the ends are strictly sorted.
pub fn validate_ends(
    ends: &ArrayRef,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    vortex_ensure!(
        ends.dtype().is_unsigned_int(),
        "run ends must be unsigned integers, was {}",
        ends.dtype(),
    );

    // Handle empty run-ends
    if ends.is_empty() {
        vortex_ensure!(
            offset == 0,
            "non-zero offset provided for empty RunEndArray"
        );
        return Ok(());
    }

    // Zero-length logical slices may retain run metadata from the source array.
    if length == 0 {
        return Ok(());
    }

    #[cfg(debug_assertions)]
    {
        // Run ends must be strictly sorted for binary search to work correctly.
        let pre_validation = ends.statistics().to_owned();

        let is_sorted = ends
            .statistics()
            .compute_is_strict_sorted(ctx)
            .unwrap_or(false);

        // Preserve the original statistics since compute_is_strict_sorted may have mutated them.
        // We don't want to run with different stats in debug mode and outside.
        ends.statistics().inherit(pre_validation.iter());
        debug_assert!(is_sorted);
    }

    // Skip host-only validation when ends are not host-resident.
    if !ends.is_host() {
        return Ok(());
    }

    // Validate the offset and length are valid for the given ends.
    if offset != 0 && length != 0 {
        let first_run_end = usize::try_from(&ends.execute_scalar(0, ctx)?)?;
        if first_run_end < offset {
            vortex_bail!("First run end {first_run_end} must be >= offset {offset}");
        }
    }

    let last_run_end = usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)?;
    let min_required_end = offset + length;
    if last_run_end < min_required_end {
        vortex_bail!("Last run end {last_run_end} must be >= offset+length {min_required_end}");
    }

    Ok(())
}

/// Compute the logical length of a run-end array from its `ends` child.
///
/// The logical length is the value of the last run end, or `0` if there are no runs.
pub fn logical_len_from_ends(ends: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    if ends.is_empty() {
        Ok(0)
    } else {
        usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)
    }
}
