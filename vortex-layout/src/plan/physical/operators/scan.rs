// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading and decoding one segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinView;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::SerializedArray;
use vortex_buffer::Buffer;
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
use crate::plan::physical::pipeline::AccessMode;
use crate::plan::physical::pipeline::sparse_selection;
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
    access_mode: AccessMode,
) -> VortexResult<ArrayRef> {
    if access_mode == AccessMode::RandomAccess {
        let rows = slice.cloned().unwrap_or(0..array.len());
        let selected = filter.map_or_else(|| rows.len(), Mask::true_count);
        let native_slice = filter.is_none_or(Mask::all_true)
            && (array.is::<Primitive>()
                || array.is::<Bool>()
                || array.is::<Decimal>()
                || array.is::<VarBinView>());
        if !native_slice && sparse_selection(selected, array.len()) {
            // Gather from the original encoding. Slicing first can hide the sparse demand from
            // kernels that compare the selected count with the encoded array's full length.
            let indices: Buffer<u64> = match filter {
                Some(Mask::Values(values)) => values
                    .indices()
                    .iter()
                    .map(|&index| (rows.start + index) as u64)
                    .collect(),
                _ => (rows.start..rows.end).map(|index| index as u64).collect(),
            };
            tracing::debug!(
                selected_rows = selected,
                total_rows = array.len(),
                "v2 sparse segment take"
            );
            return array.take(indices.into_array());
        }
    }
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
    access_mode: AccessMode,
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
        access_mode: AccessMode,
    ) -> Self {
        Self {
            unread: Some(plan.segment_id()),
            plan,
            slice,
            filter,
            access_mode,
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
            self.access_mode,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use super::*;

    #[test]
    fn contiguous_native_lookup_keeps_zero_copy_slice() -> VortexResult<()> {
        let source = PrimitiveArray::from_iter(0..1024i32).into_array();
        let actual = select(
            source,
            Some(&(7..8)),
            Some(&Mask::new_true(1)),
            AccessMode::RandomAccess,
        )?;
        assert!(actual.is::<Primitive>());
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter([7i32]),
            &mut VortexSession::empty().create_execution_ctx()
        );
        Ok(())
    }
}
