// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::plan_split;
use crate::scan::v2::io::SegmentRanges;
use crate::scan::v2::io::pump;
use crate::scan::v2::io::segment_io;
use crate::scan::v2::pool::run_on_driver_thread;
use crate::segments::SegmentSource;

/// Everything one split needs, captured when the scan is executed.
pub(super) struct SplitTask<A> {
    pub(super) plans: ScanPlans,
    pub(super) pruning: Option<PlanRef>,
    pub(super) filter: Option<PlanRef>,
    pub(super) segments: Arc<dyn SegmentSource>,
    pub(super) ranges: SegmentRanges,
    pub(super) range: Range<u64>,
    pub(super) mask: Mask,
    pub(super) map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

impl<A> SplitTask<A> {
    /// Runs the split's pruning, filter, and projection planners and its morsel on the planning
    /// driver.
    ///
    /// The driver blocks while it waits for reads, so it runs on a dedicated driver thread. This
    /// future serves its reads meanwhile, on whatever runtime drives the scan.
    pub(super) async fn run(self) -> VortexResult<Option<A>> {
        let Self {
            plans,
            pruning,
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

        let (io, reads, completions) = segment_io(ranges);
        let io = Arc::new(io);
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range,
        };
        let root = plan_split(plans, pruning, filter, scope, mask)?;
        let driver = run_on_driver_thread(move || Driver::new(io).run(root));
        let mut batches = pump(segments, reads, completions, driver).await?;
        if batches.len() > 1 {
            vortex_bail!("A split produced {} batches instead of one", batches.len());
        }
        batches.pop().map(|batch| map_fn(batch.array)).transpose()
    }
}
