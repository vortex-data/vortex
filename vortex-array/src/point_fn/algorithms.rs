// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic point-fn algorithms that build on [`PointDispatch`].

use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::cmp::Ordering::Greater;
use std::cmp::Ordering::Less;
use std::hint;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Generic binary search over an array using only `d.scalar_at`.
///
/// This is the default implementation of `SearchSortedKernel`. Encodings can override
/// `SearchSortedKernel` to push search into a child array directly (Dict, RunEnd,
/// Chunked, FoR, etc.), but for encodings without a structural shortcut this generic
/// algorithm is optimal as long as `scalar_at` is cheap.
///
/// Unlike the prior top-level `search_sorted` implementation, this function does **not**
/// construct a fresh `ExecutionCtx` per probe. It uses the dispatch's existing context,
/// so the per-probe construction cost is paid once per search rather than `log(n)` times.
pub fn generic_search_sorted<D: PointDispatch + ?Sized>(
    arr: &ArrayRef,
    target: &Scalar,
    side: SearchSortedSide,
    d: &mut D,
) -> VortexResult<SearchResult> {
    let len = arr.len();
    if len == 0 {
        return Ok(SearchResult::NotFound(0));
    }

    // Initial bracket-find pass: locate any equal element if it exists, else the
    // insertion point bracket. Adapted from Rust stdlib slice::binary_search_by, but
    // calling d.scalar_at per probe so view encodings can recurse and block-decoded
    // encodings can hit their cached block.
    let primary = search_sorted_side_idx(arr, d, 0, len, |s| {
        Ok(s.partial_cmp(target).unwrap_or(Less))
    })?;

    match primary {
        SearchResult::Found(found) => {
            // Refine for side: search the equal range for its left or right boundary.
            let (lo, hi) = match side {
                SearchSortedSide::Left => (0, found),
                SearchSortedSide::Right => (found, len),
            };
            let cmp = |s: &Scalar| -> Ordering {
                let ord = s.partial_cmp(target).unwrap_or(Less);
                match side {
                    SearchSortedSide::Left => {
                        if ord == Less {
                            Less
                        } else {
                            Greater
                        }
                    }
                    SearchSortedSide::Right => {
                        if matches!(ord, Less | Equal) {
                            Less
                        } else {
                            Greater
                        }
                    }
                }
            };
            let idx_search = search_sorted_side_idx(arr, d, lo, hi, |s| Ok(cmp(s)))?;
            Ok(match idx_search {
                SearchResult::NotFound(i) => SearchResult::Found(i),
                SearchResult::Found(_) => {
                    unreachable!("searching amongst equal values should never return Found result")
                }
            })
        }
        other => Ok(other),
    }
}

/// Branch-predictor-friendly binary search across `[from, to)` using `d.scalar_at`.
///
/// Adapted from the standard library's `slice::binary_search_by`, modified to:
/// 1. Use `d.scalar_at(arr, mid)` for probes — recurses through view encodings and hits
///    any caches the dispatch provides.
/// 2. Take an iteration-count predictable structure so CPUs can branch-predict the loop.
fn search_sorted_side_idx<D: PointDispatch + ?Sized, F>(
    arr: &ArrayRef,
    d: &mut D,
    from: usize,
    to: usize,
    mut cmp: F,
) -> VortexResult<SearchResult>
where
    F: FnMut(&Scalar) -> VortexResult<Ordering>,
{
    let mut size = to - from;
    if size == 0 {
        return Ok(SearchResult::NotFound(from));
    }
    let mut base = from;

    while size > 1 {
        let half = size / 2;
        let mid = base + half;

        let scalar = d.scalar_at(arr, mid)?;
        let ord = cmp(&scalar)?;

        // CMOV-friendly: avoid early exit on Equal so loop count depends only on size.
        base = if ord == Greater { base } else { mid };
        size -= half;
    }

    let scalar = d.scalar_at(arr, base)?;
    let ord = cmp(&scalar)?;
    if ord == Equal {
        // SAFETY: base < to by construction.
        unsafe { hint::assert_unchecked(base < to) };
        Ok(SearchResult::Found(base))
    } else {
        let result = base + (ord == Less) as usize;
        unsafe { hint::assert_unchecked(result <= to) };
        Ok(SearchResult::NotFound(result))
    }
}
