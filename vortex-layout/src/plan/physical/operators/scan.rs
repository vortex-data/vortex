// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::arrays::shared::Shared;
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
    let array = serialized.decode(
        plan.dtype(),
        usize::try_from(plan.row_count())?,
        plan.array_ctx(),
        session,
    )?;
    if array.dtype().is_variant() {
        return share_variant_storage(array, session);
    }
    Ok(array)
}

/// Shares the children of a Variant's core storage, so that each is decoded at most once however
/// many `variant_get` calls read it.
///
/// Every extraction from an unshredded path decodes the metadata and the value of the storage.
/// Sharing keeps that lazy: a query that only reads shredded children never decodes them.
fn share_variant_storage(array: ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
    let variant = array.execute::<VariantArray>(&mut session.create_execution_ctx())?;
    let core_storage = variant.core_storage();
    let slots = core_storage
        .slots()
        .iter()
        .map(|slot| {
            slot.as_ref().map(|child| {
                // A Variant child is storage of its own, not a child that decodes.
                if child.dtype().is_variant() {
                    child.clone()
                } else {
                    Shared::new(child.clone()).into_array()
                }
            })
        })
        .collect::<ArraySlots>();
    // SAFETY: a shared array has the logical values of its source.
    let core_storage = unsafe { core_storage.clone().with_slots(slots) }?;
    Ok(VariantArray::try_new(core_storage, variant.shredded().cloned())?.into_array())
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
        )?))
    }
}
