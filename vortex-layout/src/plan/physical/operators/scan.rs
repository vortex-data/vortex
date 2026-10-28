// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
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
use crate::plan::physical::pipeline::PrimitiveRead;
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
    fn primitive_read(&self) -> Option<PrimitiveRead> {
        let DType::Primitive(ptype, _) = self.plan.dtype() else {
            return None;
        };
        if !ptype.is_int() || self.plan.dtype().is_nullable() || self.plan.array_tree().is_some() {
            return None;
        }
        let width = ptype.byte_width();
        let row_count = usize::try_from(self.plan.row_count()).ok()?;
        let buffer_len = row_count.checked_mul(width)?;
        if buffer_len < 64 * 1024 {
            return None;
        }
        let slice = self.slice.clone().unwrap_or(0..row_count);
        if slice.start > slice.end
            || slice.end > row_count
            || self.filter.as_ref().is_some_and(|mask| mask.len() != slice.len())
        {
            return None;
        }
        let selected = self.filter.as_ref().map_or(slice.len(), Mask::true_count);
        if selected == 0 || selected > row_count / 8 {
            return None;
        }
        Some(PrimitiveRead::new(
            self.plan.array_ctx().clone(),
            buffer_len,
            width,
            slice.start,
            self.filter.clone().unwrap_or_else(|| Mask::new_true(slice.len())),
        ))
    }

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
            if let Some(primitive) = self.primitive_read() {
                cx.request_primitive(segment, primitive);
            } else {
                cx.request_read(segment);
            }
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
        )?))
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
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        Ok(Step::Final(select(
            batch,
            self.slice.as_ref(),
            self.filter.as_ref(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::registry::ReadContext;

    use super::ScanSource;
    use crate::plan::physical::SegmentScanPlan;
    use crate::segments::SegmentId;

    #[test]
    fn primitive_hint_does_not_materialize_sparse_ranges() -> VortexResult<()> {
        let row_count = 131072;
        let mask = Mask::from_indices(row_count, (0..row_count).step_by(16));
        let Mask::Values(values) = &mask else {
            unreachable!("the selection is neither empty nor full");
        };
        let source = ScanSource::new(
            SegmentScanPlan::new(
                PType::I64.into(),
                u64::try_from(row_count)?,
                SegmentId::from(0),
                ReadContext::new(ArrayContext::empty().to_ids()),
                None,
            ),
            None,
            Some(mask.clone()),
        );
        let _hint = source.primitive_read();
        assert!(values.cached_slices().is_none());
        Ok(())
    }
}
