// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::plan::SegmentScanPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::IoRequestId;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;

/// Reads one segment, then emits its rows as a single piece.
///
/// Two masks cover the node's rows:
///
/// - The selection is always given and says which rows the parent cares about. It is a hint: rows
///   outside it may be skipped or decoded, and the node may ignore it entirely.
/// - The optional filter says which rows to return. With a filter the piece holds exactly the
///   filter's rows, in order; without one it holds every row, dense, and values at rows outside
///   the selection are unspecified. A filter only selects rows the selection cares about.
///
/// The first compute publishes the read, unless the filter selects nothing, and the compute that
/// receives the bytes decodes them, slices them to the node's rows, and applies the filter.
pub(crate) struct SegmentScanNode {
    plan: SegmentScanPlan,
    selection: Selection,
    filter: Option<Mask>,
    state: ScanState,
}

enum ScanState {
    Init,
    Requested(IoRequestId),
    Done,
}

impl SegmentScanNode {
    pub(crate) fn try_new(
        plan: SegmentScanPlan,
        selection: Selection,
        filter: Option<Mask>,
    ) -> VortexResult<Self> {
        if let Some(filter) = &filter {
            vortex_ensure!(
                filter.len() == selection.mask().len(),
                "SegmentScan filter of length {} does not cover rows {:?}",
                filter.len(),
                selection.rows()
            );
        }
        Ok(Self {
            plan,
            selection,
            filter,
            state: ScanState::Init,
        })
    }
}

impl ExecNode for SegmentScanNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        match std::mem::replace(&mut self.state, ScanState::Done) {
            ScanState::Init if self.filter.as_ref().is_some_and(Mask::all_false) => {
                cx.emit(empty_piece(
                    self.plan.dtype(),
                    self.selection.rows().clone(),
                ));
                cx.close();
                Ok(NodeState::Done)
            }
            ScanState::Init => {
                self.state = ScanState::Requested(cx.request(self.plan.segment_id()));
                Ok(NodeState::Waiting)
            }
            ScanState::Requested(expected) => {
                let mut io = cx.take_io();
                let Some((id, segment)) = io.pop() else {
                    self.state = ScanState::Requested(expected);
                    return Ok(NodeState::Waiting);
                };
                if id != expected || !io.is_empty() {
                    vortex_bail!("SegmentScan did not expect {id:?}");
                }
                cx.emit(self.decode(segment, cx)?);
                cx.close();
                cx.yield_now();
                Ok(NodeState::Done)
            }
            ScanState::Done => vortex_bail!("SegmentScan computed after it closed"),
        }
    }
}

impl SegmentScanNode {
    /// Decodes the segment and returns its rows, filtered when the node has a filter.
    fn decode(&self, segment: BufferHandle, cx: &StepCx<'_>) -> VortexResult<Piece> {
        let serialized = match self.plan.array_tree() {
            Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?,
            None => SerializedArray::try_from(segment)?,
        };
        let row_count =
            usize::try_from(self.plan.row_count()).vortex_expect("row count must fit in usize");
        let mut array = serialized.decode(
            self.plan.dtype(),
            row_count,
            self.plan.array_ctx(),
            cx.session(),
        )?;

        let rows = self.selection.rows().clone();
        if rows.start > 0 || rows.end < self.plan.row_count() {
            let start = usize::try_from(rows.start).vortex_expect("row must fit in usize");
            let end = usize::try_from(rows.end).vortex_expect("row must fit in usize");
            array = array.slice(start..end)?;
        }
        if let Some(filter) = &self.filter
            && !filter.all_true()
        {
            array = array.filter(filter.clone())?;
        }
        Ok(Piece { rows, array })
    }
}
