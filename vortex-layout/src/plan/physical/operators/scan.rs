// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::physical::SegmentScanPlan;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
use crate::segments::SegmentId;

/// Decodes the whole of a scan's segment.
fn decode(
    plan: &SegmentScanPlan,
    segment: BufferHandle,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let serialized = match plan.array_tree() {
        Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?,
        None => SerializedArray::try_from(segment)?,
    };
    serialized.decode(
        plan.dtype(),
        usize::try_from(plan.row_count())?,
        plan.array_ctx(),
        session,
    )
}

/// The rows of a block of a segment: a page of the block-compressed encodings (Pco, Zstd), which
/// decode whole pages, and the row grid of the writer's chunks.
const SELECT_BLOCK_ROWS: usize = 8192;

/// Slices a whole decoded segment to `slice`, then keeps the rows `filter` selects.
///
/// A sparse `filter` is applied to the runs of blocks holding selected rows only, so an encoding
/// that decodes every block of the range it filters, such as Pco or Zstd, skips the blocks the
/// filter empties. The runs' rows are then concatenated into one canonical array: they are few,
/// and a consumer handed one piece per run would pay for each.
fn select(
    array: ArrayRef,
    slice: Option<&Range<usize>>,
    filter: Option<&Mask>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let first_row = slice.map_or(0, |slice| slice.start);
    let array = match slice {
        Some(slice) => array.slice(slice.clone())?,
        None => array,
    };
    let Some(filter) = filter else {
        return Ok(array);
    };
    let Some(runs) = selected_block_runs(first_row, filter) else {
        return array.filter(filter.clone());
    };
    let pieces = runs
        .into_iter()
        .map(|run| array.slice(run.clone())?.filter(filter.slice(run)))
        .collect::<VortexResult<Vec<_>>>()?;
    let mut ctx = session.create_execution_ctx();
    Ok(ChunkedArray::try_new(pieces, array.dtype().clone())?
        .into_array()
        .execute::<Canonical>(&mut ctx)?
        .into_array())
}

/// The runs of `filter`, whose first position is row `first_row` of its segment, that cover the
/// segment's [`SELECT_BLOCK_ROWS`] blocks holding a selected row, or `None` when those blocks
/// cover more than half of `filter`, so filtering it whole decodes little it does not need.
fn selected_block_runs(first_row: usize, filter: &Mask) -> Option<Vec<Range<usize>>> {
    let len = filter.len();
    let mut runs: Vec<Range<usize>> = Vec::new();
    let mut covered = 0;
    let mut start = 0;
    while start < len {
        let block = (first_row + start) / SELECT_BLOCK_ROWS;
        let end = ((block + 1) * SELECT_BLOCK_ROWS - first_row).min(len);
        if !filter.slice(start..end).all_false() {
            covered += end - start;
            match runs.last_mut() {
                Some(run) if run.end == start => run.end = end,
                _ => runs.push(start..end),
            }
        }
        start = end;
    }
    (!runs.is_empty() && covered * 2 <= len).then_some(runs)
}

/// Reads one segment, decodes it, and emits its rows of `slice`, filtered by `filter`.
pub(crate) struct ScanSource {
    plan: SegmentScanPlan,
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
    /// The segment to read, until it has been asked for.
    unread: Option<SegmentId>,
    /// The segment has been emitted.
    done: bool,
}

impl ScanSource {
    pub(crate) fn new(
        plan: SegmentScanPlan,
        slice: Option<Range<usize>>,
        filter: Option<Mask>,
    ) -> Self {
        Self {
            unread: Some(plan.segment_id()),
            plan,
            slice,
            filter,
            done: false,
        }
    }
}

impl Source for ScanSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if let Some(segment) = self.unread.take() {
            cx.request_read(segment);
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if self.done || cx.has_bytes() {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Io)
        })
    }
}

impl Operator for ScanSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        let (_, bytes) = cx
            .take_segment()
            .ok_or_else(|| vortex_err!("Segment scan ran without its bytes"))?;
        self.done = true;
        let array = decode(&self.plan, bytes, cx.session())?;
        Ok(Step::Final(select(
            array,
            self.slice.as_ref(),
            self.filter.as_ref(),
            cx.session(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_mask::Mask;

    use super::SELECT_BLOCK_ROWS;
    use super::selected_block_runs;

    const B: usize = SELECT_BLOCK_ROWS;

    #[rstest]
    #[case::dense(0, 4 * B, &[0, B, 2 * B, 3 * B], None)]
    #[case::one_block(0, 4 * B, &[B + 5], Some(vec![B..2 * B]))]
    #[case::merged_runs(0, 8 * B, &[B, 2 * B + 1, 6 * B], Some(vec![B..3 * B, 6 * B..7 * B]))]
    #[case::unaligned_slice(B / 2, 4 * B, &[1, 3 * B], Some(vec![0..B / 2, 5 * B / 2..7 * B / 2]))]
    #[case::none_selected(0, 4 * B, &[], None)]
    fn runs_cover_selected_blocks(
        #[case] first_row: usize,
        #[case] len: usize,
        #[case] selected: &[usize],
        #[case] expected: Option<Vec<Range<usize>>>,
    ) {
        let filter = Mask::from_indices(len, selected.to_vec());
        assert_eq!(selected_block_runs(first_row, &filter), expected);
    }
}
