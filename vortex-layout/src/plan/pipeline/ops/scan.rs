// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::SegmentScanPlan;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;
use crate::segments::SegmentId;

/// Decodes the whole of a scan's segment.
pub(crate) fn decode(
    plan: &SegmentScanPlan,
    segment: vortex_array::buffer::BufferHandle,
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

/// Slices a whole decoded segment to `slice`, then keeps the rows `filter` selects.
fn select(
    array: ArrayRef,
    slice: Option<&Range<usize>>,
    filter: Option<&Mask>,
) -> VortexResult<ArrayRef> {
    let array = match slice {
        Some(slice) => array.slice(slice.clone())?,
        None => array,
    };
    match filter {
        Some(filter) => array.filter(filter.clone()),
        None => Ok(array),
    }
}

enum ScanState {
    Request,
    Waiting,
    Done,
}

/// Reads one segment, decodes it, and emits its rows of `slice`, filtered by `filter`.
pub(crate) struct ScanSource {
    plan: SegmentScanPlan,
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
    state: ScanState,
}

impl ScanSource {
    pub(crate) fn new(
        plan: SegmentScanPlan,
        slice: Option<Range<usize>>,
        filter: Option<Mask>,
    ) -> Self {
        Self {
            plan,
            slice,
            filter,
            state: ScanState::Request,
        }
    }
}

impl Operator for ScanSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        match self.state {
            ScanState::Request => vortex_bail!("Segment scan computed before its read"),
            ScanState::Waiting => {
                let bytes = cx
                    .take_bytes()
                    .ok_or_else(|| vortex_err!("Segment scan ran without its bytes"))?;
                self.state = ScanState::Done;
                let array = decode(&self.plan, bytes, cx.session())?;
                Ok(Step::Last(select(
                    array,
                    self.slice.as_ref(),
                    self.filter.as_ref(),
                )?))
            }
            ScanState::Done => Ok(Step::Finished),
        }
    }
}

impl Source for ScanSource {
    fn request(&mut self) -> Option<SegmentId> {
        match self.state {
            ScanState::Request => {
                self.state = ScanState::Waiting;
                Some(self.plan.segment_id())
            }
            ScanState::Waiting | ScanState::Done => None,
        }
    }
}

/// Narrows a whole decoded segment, the one batch of a share's port, to a reader's rows.
pub(crate) struct SelectStage {
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
}

impl SelectStage {
    pub(crate) fn new(slice: Option<Range<usize>>, filter: Option<Mask>) -> Self {
        Self { slice, filter }
    }
}

impl Operator for SelectStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => Ok(Step::Last(select(
                batch,
                self.slice.as_ref(),
                self.filter.as_ref(),
            )?)),
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}
