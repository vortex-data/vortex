// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work of an opened [`PlanDataSource`](super::PlanDataSource) scan, for the scan pool.
//!
//! The units are plan work. Opening the scan prepares each ready child once, which plans without
//! reading, and runs its root up front, which spawns its parts, as develop's V2 DuckDB scan does:
//! a part is the rows between two of the child's splits for the fields the scan reads, such as a
//! file's layout splits, else one of the scan's chunks. A root whose filter changes while the scan
//! runs, as a top-N's bound does, prunes the child as a whole before it spawns parts; it reads
//! zone maps to do so, so it is an expansion instead. A worker spawns each part it takes onto the
//! session's runtime, where it is driven on a planning run of its own that awaits only its reads:
//! with a filter, the part's pruning plans first, continuing into the chunks of the rows pruning
//! kept; without, the chunks directly.
//!
//! Work creates work: expanding a deferred child, whether a file to open, a manifest entry or a
//! table partition, yields the units of the children it expands into, which may be deferred in
//! turn. The [pool](crate::pool::open_work_pool) shares the units among the workers.
//!
//! Each top-level child has an equal share of the pool's progress. A deferred child divides its
//! share evenly among the children it expands into, and a prepared child divides its share among
//! its parts by their rows.

use std::future;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::plan::DeferredPlanChild;
use super::plan::PlanChild;
use super::plan::PlanSourceChild;
use super::plan::ScanState;
use super::plan::Unit;
use crate::execution::planning::driver::Progress;
use crate::execution::planning::driver::Run;
use crate::execution::planning::planner::Planner;
use crate::execution::planning::planner::WorkScope;
use crate::pool::Expansion;
use crate::pool::PoolLimit;
use crate::pool::RowLimit;
use crate::pool::ScanPoolRef;
use crate::pool::UnitOutput;
use crate::pool::WorkPoolOptions;
use crate::pool::WorkUnit;
use crate::pool::open_work_pool;
use crate::prepare::PreparedScan;
use crate::prepare::part_root;

/// One child's prepared scan, shared by the workers running its parts.
struct PreparedChild {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
}

/// Opens a pool over the scan's `units`, preparing each of its ready children.
pub(super) fn open(state: Arc<ScanState>, units: Vec<Unit>) -> VortexResult<ScanPoolRef> {
    let share = 1.0 / units.len().max(1) as f64;
    let mut work = Vec::with_capacity(units.len());
    for unit in units {
        match state.child(unit.child) {
            PlanSourceChild::Ready(child) => {
                work.extend(prepare(&state, child, unit.rows, share)?);
            }
            PlanSourceChild::Deferred { child, .. } => {
                work.push(expand(Arc::clone(&state), Arc::clone(child), share));
            }
        }
    }
    let options = WorkPoolOptions {
        dtype: state.dtype.clone(),
        ordered: state.ordered,
        limit: state.limit.map_or(PoolLimit::Unlimited, |limit| {
            PoolLimit::Trim(RowLimit::new(limit))
        }),
        split_concurrency: state.split_concurrency,
        // An ordered pool expands each deferred child only once the units before it are taken;
        // an unordered one expands every child as soon as a worker takes it.
        expand_concurrency: if state.ordered {
            NonZeroUsize::MIN
        } else {
            NonZeroUsize::MAX
        },
        map_trimmed: None,
    };
    Ok(open_work_pool(options, work))
}

/// The unit that expands a deferred child into the units of the children it stands for.
fn expand(state: Arc<ScanState>, child: Arc<dyn DeferredPlanChild>, share: f64) -> WorkUnit {
    WorkUnit::expand(share, move || {
        async move {
            let children = child.expand().await?;
            if children.is_empty() {
                return Ok(vec![WorkUnit::skipped(share)]);
            }
            let share = share / children.len() as f64;
            let mut units = Vec::new();
            for child in children {
                match child {
                    PlanSourceChild::Ready(child) => match state.admit(child)? {
                        Some(child) => units.extend(prepare(&state, &child, None, share)?),
                        None => units.push(WorkUnit::skipped(share)),
                    },
                    PlanSourceChild::Deferred { child, .. } => {
                        units.push(expand(Arc::clone(&state), child, share));
                    }
                }
            }
            Ok(units)
        }
        .boxed()
    })
}

/// Prepares the scan of `rows` of `child`, or of the request's rows when `None`, once, and runs its
/// root up front, like the splits of a V1 scan, unless the root reads before it spawns parts.
///
/// Parts are cut on the child's splits for the fields the scan reads, when it has them, else on
/// the scan's chunks. A filter whose value changes while the scan runs, as a top-N's bound does,
/// prunes the child as a whole first, so a child it rules out takes no worker.
fn prepare(
    state: &ScanState,
    child: &PlanChild,
    rows: Option<Range<u64>>,
    share: f64,
) -> VortexResult<Vec<WorkUnit>> {
    let rows = match rows {
        Some(rows) => rows,
        None => state.child_rows(child)?,
    };
    if rows.is_empty() {
        return Ok(vec![WorkUnit::skipped(share)]);
    }
    let splits = child.splits(&state.projection, state.filter.as_ref())?;
    let scan = Arc::new(state.builder(child, rows).prepare()?);
    let dynamic = state
        .filter
        .as_ref()
        .is_some_and(|filter| DynamicExprUpdates::new(filter).is_some());
    let root = part_root(&scan, splits.as_deref(), dynamic);
    let io = scan.io();
    let child = Arc::new(PreparedChild { scan, io });
    let parts = run_root(child, root, share, state.session.clone());
    if dynamic {
        return Ok(vec![WorkUnit::expand(share, move || parts)]);
    }
    parts
        .now_or_never()
        .vortex_expect("a root that only cuts parts reads nothing")
}

/// Runs a child's root on a planning run of its own, awaiting only its reads, and returns a unit
/// for each part it spawns, in row order, dividing the child's `share` among them by their rows.
/// The rows the parts leave out are a skipped unit.
fn run_root(
    child: Arc<PreparedChild>,
    root: Box<dyn Planner>,
    share: f64,
    session: VortexSession,
) -> Expansion {
    async move {
        let rows = child.scan.rows();
        let per_row = share / (rows.end - rows.start).max(1) as f64;
        let mut unassigned = rows.end - rows.start;
        let mut run = Run::new();
        run.admit(
            root,
            WorkScope {
                file_ordinal: 0,
                rows: rows.clone(),
            },
            child.io.session(),
        );
        let mut units = Vec::new();
        loop {
            match run.advance()? {
                Progress::Spawn(scope, planner) => {
                    let part_rows = scope.rows.end - scope.rows.start;
                    unassigned = unassigned.saturating_sub(part_rows);
                    let child = Arc::clone(&child);
                    let session = session.clone();
                    units.push(WorkUnit::run(per_row * part_rows as f64, move || {
                        // Another runtime thread may run the part while this worker drives its
                        // other parts, and its reads start right away.
                        session
                            .handle()
                            .spawn(run_part(child, scope, planner))
                            .boxed()
                    }));
                }
                Progress::Batch(_) => vortex_bail!("a scan's root produced a batch"),
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        if unassigned > 0 {
            units.push(WorkUnit::skipped(per_row * unassigned as f64));
        }
        Ok(units)
    }
    .boxed()
}

/// Runs a part on a planning run of its own, awaiting only its reads, and returns its batches in
/// row order.
fn run_part(child: Arc<PreparedChild>, scope: WorkScope, planner: Box<dyn Planner>) -> UnitOutput {
    async move {
        let mut run = Run::new();
        run.admit(planner, scope, child.io.session());
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push((batch.scope.rows.start, batch.array)),
                Progress::Spawn(scope, planner) => {
                    run.admit(planner, scope, child.io.session());
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        // A part's chunks finish in whatever order their reads arrive.
        batches.sort_by_key(|(start, _)| *start);
        Ok(batches.into_iter().map(|(_, array)| array).collect())
    }
    .boxed()
}
