// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::plan::SegmentScanPlan;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;

/// Reads one segment, then emits its rows as a single array.
///
/// Two masks cover the node's rows:
///
/// - The selection is always given and says which rows the parent cares about. It is a hint: rows
///   outside it may be skipped or decoded, and the node may ignore it entirely.
/// - The optional filter says which rows to return. With a filter the array holds exactly the
///   filter's rows, in order; without one it holds every row, dense, and values at rows outside
///   the selection are unspecified. A filter only selects rows the selection cares about.
///
/// `start` publishes the read, unless the filter selects nothing or the segment is already
/// decoded, and the compute that receives the bytes decodes them, slices them to the node's
/// rows, and applies the filter.
pub(crate) struct SegmentScanNode {
    plan: SegmentScanPlan,
    selection: Selection,
    filter: Option<Mask>,
    ctx: ExecContext,
}

impl SegmentScanNode {
    pub(crate) fn try_new(
        plan: SegmentScanPlan,
        selection: Selection,
        filter: Option<Mask>,
        ctx: ExecContext,
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
            ctx,
        })
    }

    /// Decodes the whole segment.
    fn decode(&self, segment: BufferHandle) -> VortexResult<ArrayRef> {
        let serialized = match self.plan.array_tree() {
            Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?,
            None => SerializedArray::try_from(segment)?,
        };
        let row_count = usize::try_from(self.plan.row_count())?;
        serialized.decode(
            self.plan.dtype(),
            row_count,
            self.plan.array_ctx(),
            self.ctx.session(),
        )
    }

    /// Slices the whole decoded segment to the node's rows, filtered when the node has a filter.
    fn select(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        let rows = self.selection.rows();
        let mut array = if rows.start == 0 && rows.end == self.plan.row_count() {
            array
        } else {
            array.slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?
        };
        if let Some(filter) = &self.filter
            && !filter.all_true()
        {
            array = array.filter(filter.clone())?;
        }
        Ok(array)
    }
}

impl ExecNode for SegmentScanNode {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.filter.as_ref().is_some_and(Mask::all_false) {
            return Ok(NodeState::Done);
        }
        if let Some(array) = self.ctx.decoded().get(self.plan.segment_id()) {
            cx.emit(self.select(array)?);
            return Ok(NodeState::Done);
        }
        cx.request(self.plan.segment_id());
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let segment = cx
            .take_delivery()
            .ok_or_else(|| vortex_err!("SegmentScan ran without its delivery"))?;
        // Another graph sharing the cache may have decoded the segment while the read was in
        // flight; its array serves this node too.
        let array = match self.ctx.decoded().get(self.plan.segment_id()) {
            Some(array) => array,
            None => {
                let array = self.decode(segment)?;
                self.ctx
                    .decoded()
                    .insert(self.plan.segment_id(), array.clone());
                array
            }
        };
        cx.emit(self.select(array)?);
        Ok(NodeState::Done)
    }
}
