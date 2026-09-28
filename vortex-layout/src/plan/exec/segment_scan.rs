// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::SegmentScanPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::IoRequestId;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;

/// Reads one segment, then emits its selected rows as a single piece.
pub(crate) struct SegmentScanNode {
    plan: SegmentScanPlan,
    selection: Selection,
    state: ScanState,
}

enum ScanState {
    Init,
    Requested(IoRequestId),
    Loaded(BufferHandle),
    Done,
}

impl SegmentScanNode {
    pub(crate) fn new(plan: SegmentScanPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            state: ScanState::Init,
        }
    }
}

impl ExecNode for SegmentScanNode {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        if self.selection.mask().all_false() {
            cx.emit(empty_piece(
                self.plan.dtype(),
                self.selection.rows().clone(),
            ));
            cx.close();
            self.state = ScanState::Done;
        } else {
            self.state = ScanState::Requested(cx.request(self.plan.segment_id()));
        }
        Ok(())
    }

    fn is_ready(&self) -> bool {
        matches!(self.state, ScanState::Loaded(_))
    }

    fn on_io(&mut self, id: IoRequestId, result: BufferHandle) -> VortexResult<()> {
        match self.state {
            ScanState::Requested(expected) if expected == id => {
                self.state = ScanState::Loaded(result);
                Ok(())
            }
            _ => vortex_bail!("SegmentScan did not expect {id:?}"),
        }
    }

    fn step(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let ScanState::Loaded(segment) = std::mem::replace(&mut self.state, ScanState::Done) else {
            vortex_bail!("SegmentScan stepped before its segment arrived");
        };
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
        if !self.selection.mask().all_true() {
            array = array.filter(self.selection.mask().clone())?;
        }

        cx.emit(Piece { rows, array });
        cx.close();
        cx.yield_now();
        Ok(())
    }
}
