// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoSource;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::scan::planning::FilterPlanner;
use crate::scan::planning::ProjectionMorsel;
use crate::scan::planning::ProjectionPlanner;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::plan_split;
use crate::scan::v2::io::SegmentIoSource;
use crate::scan::v2::io::SegmentRanges;
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
    /// The driver runs inside this future, on whichever thread polls it, and the future awaits
    /// its reads between steps. Nothing is handed to another thread.
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

        let io = Arc::new(SegmentIoSource::new(segments, ranges));
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range,
        };
        let root = plan_split(plans, pruning, filter, scope, mask)?;
        let mut run = SplitRun(Driver::new(Arc::clone(&io) as Arc<dyn IoSource>).start(root));
        let mut batches = loop {
            match run.0.advance()? {
                Progress::Done(batches) => break batches,
                Progress::Waiting => {
                    let completion = io.next_completion().await?;
                    run.0.complete(completion)?;
                }
            }
        };
        if batches.len() > 1 {
            vortex_bail!("A split produced {} batches instead of one", batches.len());
        }
        batches.pop().map(|batch| map_fn(batch.array)).transpose()
    }
}

/// A split's driver run, carried by the split's future across awaits.
struct SplitRun(Run);

// SAFETY: A `Run` is not `Send` only because the protocol lets planners and morsels hold
// thread-local state, and it stores them as `Box<dyn Planner>` and `Box<dyn Morsel>`. The run of a
// split only ever holds what `plan_split` builds, the `FilterPlanner`, `ProjectionPlanner` and
// `ProjectionMorsel` asserted `Send` below, and it is owned by one future and never shared, so
// moving it between threads with that future is sound. Any planner `plan_split` gains must be
// added to the assertion.
unsafe impl Send for SplitRun {}

const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<FilterPlanner>();
    assert_send::<ProjectionPlanner>();
    assert_send::<ProjectionMorsel>();
};
