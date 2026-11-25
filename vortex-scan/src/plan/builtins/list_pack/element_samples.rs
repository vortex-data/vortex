// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

/// A sparse index from the element rows of a list column to the lists that hold them.
///
/// It records the list holding every `stride`-th element row: list `i` holds element row
/// `k * stride` exactly when it is the last list starting at or before that row. With it, a
/// selection of lists maps to the element rows that hold them, to within `stride` elements on
/// either side, without reading the offsets. Sampling the elements rather than the lists bounds
/// what a selection reads in elements, whatever the lists' lengths.
///
/// The samples are plain plan data and say nothing about how they are stored. A storage format
/// records them however it likes and rebuilds them with [`ElementSamples::try_new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementSamples {
    stride: u64,
    lists: Arc<[u64]>,
}

impl ElementSamples {
    /// Samples every `stride`-th of `element_rows` elements of `list_rows` lists.
    ///
    /// There must be one list per sample, `element_rows.div_ceil(stride)` in all, never
    /// decreasing, and each less than `list_rows`.
    pub fn try_new(
        stride: u64,
        lists: impl Into<Arc<[u64]>>,
        list_rows: u64,
        element_rows: u64,
    ) -> VortexResult<Self> {
        let lists = lists.into();
        vortex_ensure!(stride > 0, "List element samples need a nonzero stride");
        vortex_ensure_eq!(
            lists.len() as u64,
            element_rows.div_ceil(stride),
            "List element samples must cover every stride of the elements"
        );
        vortex_ensure!(
            lists.is_sorted(),
            "List element samples must never decrease"
        );
        if let Some(&last) = lists.last() {
            vortex_ensure!(
                last < list_rows,
                "List element sample {last} is not one of the {list_rows} lists"
            );
        }
        Ok(Self { stride, lists })
    }

    /// The element rows between consecutive samples.
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// The list holding element row `k * stride`, for each `k`.
    pub fn lists(&self) -> &[u64] {
        &self.lists
    }

    /// The element rows holding `lists` in a column of `element_rows` elements: every element
    /// of those lists, and at most `stride` more on either side.
    pub fn element_range(&self, lists: Range<u64>, element_rows: u64) -> Range<u64> {
        // Sample `k` is at or before the first list's first element exactly when an earlier
        // list holds it, and before the last list's end exactly when a list before the end
        // holds it.
        let after_start = self.lists.partition_point(|&list| list < lists.start) as u64;
        let through_end = self.lists.partition_point(|&list| list < lists.end) as u64;
        if through_end == 0 {
            // No list before the end holds an element.
            return 0..0;
        }
        after_start.saturating_sub(1) * self.stride..(through_end * self.stride).min(element_rows)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::ElementSamples;

    /// Lists of these lengths, with empty lists at the ends and in runs.
    const LENGTHS: [u64; 16] = [0, 0, 5, 1, 0, 40, 3, 0, 0, 7, 1, 1, 0, 64, 2, 0];

    /// Every run of lists maps to element rows holding all of their elements, and at most a
    /// stride more on either side.
    #[rstest]
    fn element_range_bounds_the_elements_of_any_lists(
        #[values(1, 3, 8, 16, 1_000)] stride: u64,
    ) -> VortexResult<()> {
        let mut offsets = vec![0u64];
        for length in LENGTHS {
            offsets.push(offsets[offsets.len() - 1] + length);
        }
        let element_rows = offsets[offsets.len() - 1];
        let list_rows = LENGTHS.len() as u64;
        // The list holding each sampled element: the last list starting at or before it.
        let lists: Vec<u64> = (0..element_rows)
            .step_by(usize::try_from(stride)?)
            .map(|row| offsets[..LENGTHS.len()].partition_point(|&start| start <= row) as u64 - 1)
            .collect();
        let samples = ElementSamples::try_new(stride, lists, list_rows, element_rows)?;
        for start in 0..list_rows {
            for end in start..=list_rows {
                let (first, last) = (
                    offsets[usize::try_from(start)?],
                    offsets[usize::try_from(end)?],
                );
                let range = samples.element_range(start..end, element_rows);
                assert!(
                    range.start <= first && last <= range.end,
                    "{start}..{end} @ {stride}"
                );
                if first < last {
                    assert!(first <= range.start + stride && range.end < last + stride);
                }
            }
        }
        Ok(())
    }

    #[rstest]
    #[case::zero_stride(0, vec![], 1, 0, "nonzero stride")]
    #[case::too_few(2, vec![0], 1, 3, "cover every stride")]
    #[case::too_many(2, vec![0, 0, 0], 1, 3, "cover every stride")]
    #[case::decreasing(1, vec![1, 0], 2, 2, "never decrease")]
    #[case::past_the_lists(1, vec![0, 2], 2, 2, "is not one of the 2 lists")]
    fn try_new_rejects_inconsistent_samples(
        #[case] stride: u64,
        #[case] lists: Vec<u64>,
        #[case] list_rows: u64,
        #[case] element_rows: u64,
        #[case] message: &str,
    ) {
        let err = ElementSamples::try_new(stride, lists, list_rows, element_rows)
            .expect_err("inconsistent samples must be rejected");
        assert!(err.to_string().contains(message), "{err}");
    }
}
