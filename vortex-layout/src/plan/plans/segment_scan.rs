// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use vortex_array::EmptyMetadata;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::segments::SegmentId;

/// Reads one serialized array segment.
#[derive(Clone, Debug)]
pub struct SegmentScan;

/// Data needed to read and decode a single segment.
#[derive(Clone, Debug)]
pub struct SegmentScanData {
    segment_id: SegmentId,
    array_ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
}

/// A plan that reads one serialized array segment.
pub type SegmentScanPlan = Plan<SegmentScan>;

impl SegmentScanPlan {
    /// Creates a segment scan over `segment_id`.
    pub fn new(
        dtype: DType,
        row_count: u64,
        segment_id: SegmentId,
        array_ctx: ReadContext,
        array_tree: Option<ByteBuffer>,
    ) -> Self {
        PlanParts {
            vtable: SegmentScan,
            dtype,
            row_count,
            children: PlanChildren::default(),
            data: SegmentScanData {
                segment_id,
                array_ctx,
                array_tree,
            },
        }
        .into_typed()
    }

    /// Returns the segment this plan reads.
    pub fn segment_id(&self) -> SegmentId {
        self.data().segment_id
    }

    /// Returns the read context for the serialized array.
    pub fn array_ctx(&self) -> &ReadContext {
        &self.data().array_ctx
    }

    /// Returns the serialized array encoding tree, when it is stored out of line.
    pub fn array_tree(&self) -> Option<&ByteBuffer> {
        self.data().array_tree.as_ref()
    }
}

impl PlanVTable for SegmentScan {
    type PlanData = SegmentScanData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.segment_scan");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The segment ID and read context are not yet covered by a metadata codec.
        None
    }

    fn with_children(
        _plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("SegmentScan", children, 0)?;
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= plan.row_count(),
            "SegmentScan row range {:?} is outside 0..{}",
            row_range,
            plan.row_count()
        );
        let row_count = usize::try_from(plan.row_count())?;
        let row_range = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
        vortex_ensure!(
            mask.len() == row_range.len(),
            "SegmentScan mask length mismatch"
        );

        let segment = ctx.segment_source().request(plan.segment_id());
        let array_ctx = plan.array_ctx().clone();
        let array_tree = plan.array_tree().cloned();
        let dtype = plan.dtype().clone();
        let session = ctx.session().clone();

        Ok(async move {
            let segment = segment.await?;
            let serialized = if let Some(array_tree) = array_tree {
                SerializedArray::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                SerializedArray::try_from(segment)?
            };
            let mut array = serialized.decode(&dtype, row_count, &array_ctx, &session)?;
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range)?;
            }
            let mask = mask.await?;
            if !mask.all_true() {
                array = array.filter(mask)?;
            }
            Ok(array)
        }
        .boxed())
    }
}
