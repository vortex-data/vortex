// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_session::VortexSession;

use crate::plan::SegmentScanPlan;
use crate::plan::exec::Event;
use crate::plan::exec::ExecContext;
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
    ctx: ExecContext,
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
                Ok(NodeState::Done)
            }
            ScanState::Init => {
                if let Some(array) = self.ctx.decoded().get(self.plan.segment_id()) {
                    cx.emit(self.select(array)?);
                    return Ok(NodeState::Done);
                }
                self.state = ScanState::Requested(cx.request(self.plan.segment_id()));
                Ok(NodeState::Wait)
            }
            ScanState::Requested(expected) => {
                let mut events = cx.events();
                let Some(event) = events.next_back() else {
                    self.state = ScanState::Requested(expected);
                    return Ok(NodeState::Wait);
                };
                let Event::Delivered(id, segment) = event else {
                    return Err(event.unexpected("SegmentScan"));
                };
                if id != expected || events.len() != 0 {
                    vortex_bail!("SegmentScan did not expect {id:?}");
                }
                let array = self.decode(segment)?;
                self.ctx
                    .decoded()
                    .insert(self.plan.segment_id(), array.clone());
                cx.emit(self.select(array)?);
                Ok(NodeState::Done)
            }
            ScanState::Done => vortex_bail!("SegmentScan computed after it closed"),
        }
    }
}

impl SegmentScanNode {
    /// Decodes the whole segment.
    fn decode(&self, segment: BufferHandle) -> VortexResult<ArrayRef> {
        decode_segment(&self.plan, self.ctx.session(), segment)
    }

    /// Slices the whole decoded segment to the node's rows, filtered when the node has a filter.
    fn select(&self, array: ArrayRef) -> VortexResult<Piece> {
        let rows = self.selection.rows().clone();
        let mut array = slice_rows(&self.plan, array, &rows)?;
        if let Some(filter) = &self.filter
            && !filter.all_true()
        {
            array = array.filter(filter.clone())?;
        }
        Ok(Piece { rows, array })
    }
}

/// Decodes the whole segment `plan` reads from its bytes.
pub(crate) fn decode_segment(
    plan: &SegmentScanPlan,
    session: &VortexSession,
    segment: BufferHandle,
) -> VortexResult<ArrayRef> {
    let serialized = match plan.array_tree() {
        Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?,
        None => SerializedArray::try_from(segment)?,
    };
    let row_count = usize::try_from(plan.row_count()).vortex_expect("row count must fit in usize");
    let array = serialized.decode(plan.dtype(), row_count, plan.array_ctx(), session)?;
    let Some(runend) = array.as_opt::<RunEnd>() else {
        return Ok(array);
    };
    if runend.ends().is_canonical() {
        return Ok(array);
    }
    if !runend
        .ends()
        .depth_first_traversal()
        .any(|array| array.is::<Pco>())
    {
        return Ok(array);
    }
    // PCO scalar probes decompress pages, which slicing repeats during binary search.
    // Other encodings can probe cheaply; eagerly decoding their whole index wastes work.
    let mut ctx = session.create_execution_ctx();
    let ends = runend.ends().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let prepared = RunEnd::try_new_offset_length(
        ends.into_array(),
        runend.values().clone(),
        runend.offset(),
        array.len(),
        &mut ctx,
    )?
    .into_array();
    prepared
        .statistics()
        .inherit(array.statistics().to_owned().iter());
    Ok(prepared)
}

/// Slices the whole decoded segment of `plan` to `rows`.
pub(crate) fn slice_rows(
    plan: &SegmentScanPlan,
    array: ArrayRef,
    rows: &Range<u64>,
) -> VortexResult<ArrayRef> {
    if rows.start == 0 && rows.end == plan.row_count() {
        return Ok(array);
    }
    let start = usize::try_from(rows.start).vortex_expect("row must fit in usize");
    let end = usize::try_from(rows.end).vortex_expect("row must fit in usize");
    array.slice(start..end)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::DictArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::segments::SegmentId;

    #[rstest]
    #[case::whole(0, 15)]
    #[case::offset(2, 11)]
    #[case::null_run(5, 4)]
    #[case::empty(0, 0)]
    fn compressed_run_ends_preserve_offsets_and_nulls(
        #[case] offset: usize,
        #[case] len: usize,
        #[values(false, true)] pco: bool,
    ) -> VortexResult<()> {
        let session = crate::test::new_session();
        vortex_runend::initialize(&session);
        session.arrays().register(Pco);
        let mut ctx = session.create_execution_ctx();
        let ends = if pco {
            let ends = PrimitiveArray::new(Buffer::from(vec![5_u32, 9, 15]), Validity::NonNullable);
            Pco::from_primitive(ends.as_view(), 0, 128, &mut ctx)?.into_array()
        } else {
            DictArray::try_new(
                Buffer::from(vec![0_u8, 1, 2]).into_array(),
                Buffer::from(vec![5_u32, 9, 15]).into_array(),
            )?
            .into_array()
        };
        let values = PrimitiveArray::from_option_iter([Some(10_i32), None, Some(30)]).into_array();
        let array =
            RunEnd::try_new_offset_length(ends, values, offset, len, &mut ctx)?.into_array();
        let array_ctx = ArrayContext::empty();
        let mut bytes = ByteBufferMut::empty_aligned(Alignment::new(64));
        for buffer in array.serialize(
            &array_ctx,
            &session,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )? {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let plan = SegmentScanPlan::new(
            array.dtype().clone(),
            len as u64,
            SegmentId::from(0),
            ReadContext::new(array_ctx.to_ids()),
            None,
        );
        let decoded = decode_segment(&plan, &session, BufferHandle::new_host(bytes.freeze()))?;
        let expected = PrimitiveArray::from_option_iter(
            [Some(10_i32); 5]
                .into_iter()
                .chain([None; 4])
                .chain([Some(30); 6]),
        )
        .into_array()
        .slice(offset..offset + len)?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
