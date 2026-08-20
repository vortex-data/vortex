// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::plan::Plan;
use crate::plan::PlanChildren;
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
}
