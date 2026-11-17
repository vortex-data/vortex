// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod execute;

use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::execution::pipeline::Chain;
use vortex_scan::execution::pipeline::Compiler;
use vortex_scan::plan::Plan;
use vortex_scan::plan::PlanId;
use vortex_scan::plan::PlanParts;
use vortex_scan::plan::PlanSlots;
use vortex_scan::plan::PlanVTable;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use self::execute::ScanSource;
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

impl SegmentScan {
    /// Creates a segment scan over `segment_id`.
    pub fn new_plan(
        dtype: DType,
        row_count: u64,
        segment_id: SegmentId,
        array_ctx: ReadContext,
        array_tree: Option<ByteBuffer>,
    ) -> SegmentScanPlan {
        PlanParts {
            vtable: SegmentScan,
            dtype,
            row_count,
            slots: PlanSlots::default(),
            data: SegmentScanData {
                segment_id,
                array_ctx,
                array_tree,
            },
        }
        .into_typed()
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

    fn with_slots(
        _plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("SegmentScan", 0)?;
        Ok(())
    }

    fn is_data_source(_plan: &Plan<Self>) -> bool {
        true
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        // A scan keeps the selected rows itself, as every plan produces only those.
        if mask.all_false() {
            return Ok(None);
        }
        let filter = (!mask.all_true()).then(|| mask.clone());
        let slice = if rows.start == 0 && rows.end == plan.row_count() {
            None
        } else {
            Some(usize::try_from(rows.start)?..usize::try_from(rows.end)?)
        };
        compiler.register_segment(plan.data().segment_id);
        Ok(Some(Chain::new(ScanSource::new(
            plan.clone(),
            slice,
            filter,
        ))))
    }
}
