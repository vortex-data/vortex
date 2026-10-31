// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::optimizer::ArrayOptimizer;
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

/// Slices a whole decoded segment to `slice`, then keeps the rows `filter` selects.
///
/// The result is optimized with `session`, so the stages that read the batch see the encoding
/// the slice and filter reduce to rather than lazy wrappers around the decoded array.
fn select(
    array: ArrayRef,
    slice: Option<&Range<usize>>,
    filter: Option<&Mask>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let array = match slice {
        Some(slice) => array.slice(slice.clone())?,
        None => array,
    };

    let array = match filter {
        Some(filter) => array.filter(filter.clone())?,
        None => array,
    };

    array.optimize(session)
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
