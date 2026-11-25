// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The parts of a prepared scan, the units a host's threads take, and the planner of each.
//!
//! A scan's parts are cut by [`split_points`](crate::split::split_points), which both engines
//! share, over the chunk boundaries [`chunk_boundaries`] finds in the scan's plans.

use std::future;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_mask::Mask;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use crate::plan::PlanRef;
use crate::plan::builtins::Concat;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// The sorted chunk boundaries of the columns `scan` reads, in the file's row domain: the row
/// offsets of every [`Concat`] in its projection and filter plans that spans the file's rows.
///
/// A plan's children are searched where they span the same rows as it, so zone tables and
/// dictionary values are skipped. The children of a [`Concat`] are not built: they are kept lazy
/// until a chunk reads them, so a column chunked within its chunks contributes only its outer
/// boundaries.
pub(crate) fn chunk_boundaries(scan: &PreparedScan) -> VortexResult<Vec<u64>> {
    let mut boundaries = Vec::new();
    concat_offsets(&scan.projection, &mut boundaries)?;
    if let Some(filter) = &scan.filter {
        for predicate in filter.predicates() {
            concat_offsets(predicate, &mut boundaries)?;
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    Ok(boundaries)
}

fn concat_offsets(plan: &PlanRef, boundaries: &mut Vec<u64>) -> VortexResult<()> {
    if let Some(concat) = plan.as_opt::<Concat>() {
        boundaries.extend_from_slice(concat.row_offsets());
        return Ok(());
    }
    for child in plan.children() {
        let child = child?;
        if child.row_count() == plan.row_count() {
            concat_offsets(child, boundaries)?;
        }
    }
    Ok(())
}

/// The rows of `scan` its filter's static conjuncts keep, by the zones their pruning plans keep,
/// or `None` when no static conjunct has a pruning plan.
///
/// The plans run once over all of the scan's rows, reading only zone maps, so the scan can be cut
/// into parts by the rows that survive them and each part skips their zones without pruning
/// again.
pub(crate) async fn static_kept(
    scan: &Arc<PreparedScan>,
    io: &Arc<dyn IoService>,
) -> VortexResult<Option<Mask>> {
    let pruning: Vec<PlanRef> = scan
        .filter
        .iter()
        .flat_map(|filter| filter.static_pruning().cloned())
        .collect();
    if pruning.is_empty() {
        return Ok(None);
    }
    let rows = scan.rows.clone();
    let len = usize::try_from(rows.end - rows.start)?;
    // Rows every plan prunes end the planner before it continues, leaving none kept.
    let kept = Arc::new(Mutex::new(Mask::new_false(len)));
    let sink = Arc::clone(&kept);
    let planner = PrunePlanner::over_kept(
        Arc::clone(scan),
        rows.clone(),
        pruning,
        None,
        Box::new(move |mask| {
            *sink.lock() = mask.unwrap_or_else(|| Mask::new_true(len));
            Box::new(Finished)
        }),
    );
    let mut run = Run::new();
    run.admit(
        Box::new(planner),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    while run.live_roots() > 0 {
        match run.advance()? {
            Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            Progress::Idle => break,
            Progress::Batch(_) | Progress::Spawn(..) | Progress::RootDone(_) => {}
        }
    }
    let kept = kept.lock().clone();
    Ok(Some(kept))
}

/// A planner with nothing to do.
struct Finished;

impl IoConsumer for Finished {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Finished {
    fn state(&self) -> State {
        State::Done
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Done)
    }
}

/// The planner of one part of `scan`, of which `kept` is the rows the scan's static conjuncts
/// keep, from [`static_kept`].
///
/// The part plans only those rows. With a dynamic conjunct, such as the bound a top-N pushes
/// down, the part first prunes them by its zones on the thread that takes it, from the zone maps
/// the file's cache holds, as the conjunct stands then, so later parts skip more as it tightens.
pub(crate) fn part_planner(
    scan: &Arc<PreparedScan>,
    part: Range<u64>,
    kept: Option<Mask>,
) -> Box<dyn Planner> {
    let chunks = {
        let scan = Arc::clone(scan);
        let part = part.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, part, selection, kept))
        }
    };
    match scan.filter() {
        Some(filter) => Box::new(PrunePlanner::over_kept(
            Arc::clone(scan),
            part,
            filter.dynamic_pruning().cloned().collect(),
            kept,
            Box::new(chunks),
        )),
        None => chunks(kept),
    }
}
