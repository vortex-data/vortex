// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoBatch;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoSource;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::PlanRef;
use crate::scan::planning::FilterPlanner;
use crate::scan::planning::FilterPlans;
use crate::scan::planning::ProjectionMorsel;
use crate::scan::planning::ProjectionPlanner;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::plan_split;
use crate::scan::v2::io::ScanIo;

/// Everything one filter split needs, captured when the scan is executed.
pub(super) struct SplitTask {
    pub(super) plans: ScanPlans,
    pub(super) pruning: Option<PlanRef>,
    pub(super) filter: Option<FilterPlans>,
    pub(super) io: Arc<dyn ScanIo>,
    /// Announcements of the segments the split is likely to read.
    pub(super) announce: IoBatch,
    pub(super) range: Range<u64>,
    pub(super) mask: Mask,
}

impl SplitTask {
    /// Runs the filter split's pruning, filter, and projection planners and its projection
    /// morsels on the planning driver, and returns one batch per projection split with selected
    /// rows, in row order.
    ///
    /// The driver runs inside this future, on whichever thread polls it, and the future awaits
    /// its reads between steps. Nothing is handed to another thread.
    pub(super) async fn run(self) -> VortexResult<Vec<ArrayRef>> {
        let Self {
            plans,
            pruning,
            filter,
            io,
            announce,
            range,
            mask,
        } = self;
        if mask.all_false() {
            return Ok(Vec::new());
        }

        let io = io.split_io();
        if !announce.is_empty() {
            io.submit(ANNOUNCER, announce)?;
        }
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
                    let completion = future::poll_fn(|cx| io.poll_completion(cx)).await?;
                    run.0.complete(completion)?;
                }
            }
        };
        // Morsels finish in whatever order their reads arrive.
        batches.sort_by_key(|batch| batch.scope.rows.start);
        Ok(batches.into_iter().map(|batch| batch.array).collect())
    }
}

/// The owner of a split's announcements, which no driver work item uses: the driver numbers its
/// work from zero.
const ANNOUNCER: IoOwnerId = IoOwnerId(u64::MAX);

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
