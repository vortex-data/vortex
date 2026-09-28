// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::plan_split;
use crate::scan::v2::io::SegmentIoSource;
use crate::scan::v2::io::SegmentRanges;
use crate::segments::SegmentSource;

/// Everything one split needs, captured when the scan is executed.
pub(super) struct SplitTask<A> {
    pub(super) plans: ScanPlans,
    pub(super) filter: Option<PlanRef>,
    pub(super) segments: Arc<dyn SegmentSource>,
    pub(super) ranges: SegmentRanges,
    pub(super) range: Range<u64>,
    pub(super) mask: Mask,
    pub(super) map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

impl<A> SplitTask<A> {
    /// Runs the split's filter planner, projection planner, and morsel on the planning driver.
    ///
    /// The driver blocks while it waits for reads, so it runs on the runtime's blocking pool
    /// while the reads themselves run on the runtime.
    pub(super) async fn run(self) -> VortexResult<Option<A>> {
        let Self {
            plans,
            filter,
            segments,
            ranges,
            range,
            mask,
            map_fn,
        } = self;
        if mask.all_false() {
            return Ok(None);
        }

        let handle = plans.session.handle();
        let io = Arc::new(SegmentIoSource::new(segments, ranges, handle.clone()));
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range,
        };
        let root = plan_split(plans, filter, scope, mask)?;
        let mut batches = handle
            .spawn_blocking(move || Driver::new(io).run(root))
            .await?;
        if batches.len() > 1 {
            vortex_bail!("A split produced {} batches instead of one", batches.len());
        }
        batches.pop().map(|batch| map_fn(batch.array)).transpose()
    }
}
