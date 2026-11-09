// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::PointReads;
use crate::plan::physical::SegmentScanPlan;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;

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

/// What a [`ScanSource`] waits for next.
enum Phase {
    /// Nothing has been asked for.
    Unread,
    /// The whole segment.
    Whole,
    /// The point reads, by their index in [`PointReads::reads`], as they arrive.
    Points(Vec<Option<BufferHandle>>),
    /// The point reads have all arrived and the segment's tree confirms them.
    Confirmed(Vec<BufferHandle>),
    /// The rows have been emitted.
    Done,
}

/// Reads one segment, decodes it, and emits its rows of `slice`, filtered by `filter`.
///
/// When the rows are a few fixed-width values of an uncompressed segment, it reads only their
/// bytes. See [`PointReads`].
pub(crate) struct ScanSource {
    plan: SegmentScanPlan,
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
    points: Option<PointReads>,
    phase: Phase,
}

impl ScanSource {
    pub(crate) fn new(
        plan: SegmentScanPlan,
        slice: Option<Range<usize>>,
        filter: Option<Mask>,
        points: Option<PointReads>,
    ) -> Self {
        Self {
            plan,
            slice,
            filter,
            points,
            phase: Phase::Unread,
        }
    }

    /// Takes the point reads that have arrived. Once all have, the reads are confirmed, or the
    /// whole segment is asked for in their place.
    fn take_points(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        let (Phase::Points(arrived), Some(points)) = (&mut self.phase, &self.points) else {
            vortex_bail!("A segment scan took point reads it did not ask for");
        };

        while let Some((read, bytes)) = cx.take_read() {
            let index = points
                .reads()
                .position(|asked| asked == read.bytes)
                .ok_or_else(|| vortex_err!("A segment scan got bytes it did not ask for"))?;
            arrived[index] = Some(bytes);
        }
        if arrived.iter().any(Option::is_none) {
            return Ok(SourceReady::Blocked(SourceBlocked::Io));
        }

        let mut arrived = std::mem::take(arrived).into_iter().flatten();
        let tail = points.guessed().then(|| arrived.next()).flatten();
        if points.confirms(tail.as_ref(), self.plan.array_ctx()) {
            self.phase = Phase::Confirmed(arrived.collect());
            return Ok(SourceReady::Ready);
        }

        cx.request_read(self.plan.segment_id());
        self.phase = Phase::Whole;
        Ok(SourceReady::Blocked(SourceBlocked::Io))
    }
}

impl Source for ScanSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        match &self.phase {
            Phase::Unread => {
                let segment = self.plan.segment_id();
                match &self.points {
                    Some(points) => {
                        for bytes in points.reads() {
                            cx.request_bytes(segment, bytes);
                        }
                        self.phase = Phase::Points(vec![None; points.reads().count()]);
                    }
                    None => {
                        cx.request_read(segment);
                        self.phase = Phase::Whole;
                    }
                }
                Ok(SourceReady::Blocked(SourceBlocked::Io))
            }
            Phase::Whole => Ok(if cx.has_bytes() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Io)
            }),
            Phase::Points(_) => self.take_points(cx),
            Phase::Confirmed(_) | Phase::Done => Ok(SourceReady::Ready),
        }
    }
}

impl Operator for ScanSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let array = match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Done => return Ok(Step::End),
            Phase::Confirmed(runs) => self
                .points
                .as_ref()
                .ok_or_else(|| vortex_err!("A segment scan confirmed no point reads"))?
                .assemble(runs)?,
            Phase::Whole => {
                let (_, bytes) = cx
                    .take_segment()
                    .ok_or_else(|| vortex_err!("Segment scan ran without its bytes"))?;
                let array = decode(&self.plan, bytes, cx.session())?;
                select(array, self.slice.as_ref(), self.filter.as_ref())?
            }
            Phase::Unread | Phase::Points(_) => {
                vortex_bail!("A segment scan computed before its bytes arrived")
            }
        };
        Ok(Step::Final(array))
    }
}
