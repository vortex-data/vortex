// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
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
use crate::plan::physical::pipeline::RetainedArrays;
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
pub(crate) fn select(
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

/// Reads one segment, decodes it, and emits its rows of `slice`, filtered by `filter`.
pub(crate) struct ScanSource {
    plan: SegmentScanPlan,
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
    /// The segment to read, until it has been asked for.
    unread: Option<SegmentId>,
    /// The segment has been emitted.
    done: bool,
    /// Where the decoded segment is kept for the file's other scans, with the file rows it
    /// covers.
    retain: Option<(RetainedArrays, Range<u64>)>,
    /// The decoded segment, when another chunk retained it before this one asked for its bytes.
    retained: Option<ArrayRef>,
}

impl ScanSource {
    pub(crate) fn new(
        plan: SegmentScanPlan,
        slice: Option<Range<usize>>,
        filter: Option<Mask>,
        retain: Option<(RetainedArrays, Range<u64>)>,
    ) -> Self {
        Self {
            unread: Some(plan.segment_id()),
            plan,
            slice,
            filter,
            done: false,
            retain,
            retained: None,
        }
    }
}

impl Source for ScanSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if let Some(segment) = self.unread.take() {
            // A neighbouring chunk may have decoded the segment since this one was compiled.
            if let Some((retained, _)) = &self.retain
                && let Some(array) = retained.get(segment)
            {
                self.retained = Some(array);
                return Ok(SourceReady::Ready);
            }
            cx.request_read(segment);
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }
        Ok(if self.done || self.retained.is_some() || cx.has_bytes() {
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
        if let Some(array) = self.retained.take() {
            self.done = true;
            return Ok(Step::Final(select(
                array,
                self.slice.as_ref(),
                self.filter.as_ref(),
            )?));
        }
        let (_, bytes) = cx
            .take_segment()
            .ok_or_else(|| vortex_err!("Segment scan ran without its bytes"))?;
        self.done = true;
        let len = bytes.len() as u64;
        let array = decode(&self.plan, bytes, cx.session())?;
        if let Some((retained, rows)) = self.retain.take() {
            retained.insert(self.plan.segment_id(), &array, len, rows);
        }
        Ok(Step::Final(select(
            array,
            self.slice.as_ref(),
            self.filter.as_ref(),
        )?))
    }
}
