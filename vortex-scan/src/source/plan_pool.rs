// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The work of an opened [`PlanDataSource`](super::PlanDataSource) scan, for the scan pool.
//!
//! The units are plan work. Opening the scan prepares each ready child once, which plans without
//! reading, and runs its root up front, which spawns its parts, as develop's V2 DuckDB scan does:
//! a part is the rows between two of the child's splits for the fields the scan reads, such as a
//! file's layout splits, else one of the scan's chunks. A root whose filter changes while the scan
//! runs, as a top-N's bound does, prunes the child as a whole before it spawns parts; it reads
//! zone maps to do so, so it is an expansion instead. A part is driven on a planning run of its own
//! that awaits only its reads: with a filter, the part's pruning plans first, continuing into the
//! chunks of the rows pruning kept; without, the chunks directly. An unordered pool's lane runs the
//! part itself and passes each chunk on as it finishes, as a DuckDB thread of develop's V2 scan
//! does, so a top-N sees rows, and tightens its bound, before the part is done. An ordered pool's
//! worker spawns each part onto the session's runtime and returns its chunks in row order.
//!
//! Work creates work: expanding a deferred child, whether a file to open, a manifest entry or a
//! table partition, yields the units of the children it expands into, which may be deferred in
//! turn. The [pool](crate::pool::open_work_pool) runs the units on its lanes and shares their output among the
//! workers.
//!
//! Each top-level child has an equal share of the pool's progress. A deferred child divides its
//! share evenly among the children it expands into, and a prepared child divides its share among
//! its parts by their rows.

use std::future;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::stream;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoService;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

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
use crate::pool::UnitStream;
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
        handle: state.session.handle(),
        // An ordered pool expands each deferred child only once the units before it are taken;
        // an unordered one opens as many children at once as V1 opens files, one per core, so the
        // parts of the first children run before every footer is read.
        expand_concurrency: if state.ordered {
            NonZeroUsize::MIN
        } else {
            get_available_parallelism()
                .and_then(NonZeroUsize::new)
                .unwrap_or(NonZeroUsize::MIN)
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
    let parts = run_root(child, root, share, state.ordered, state.session.clone());
    if dynamic {
        return Ok(vec![WorkUnit::expand(share, move || parts)]);
    }
    parts
        .now_or_never()
        .vortex_expect("a root that only cuts parts reads nothing")
}

/// Runs a child's root on a planning run of its own, awaiting only its reads, and returns a unit
/// for each part it spawns, in row order, as one turn, dividing the child's `share` among them by
/// their rows. The rows the parts leave out are a skipped unit.
fn run_root(
    child: Arc<PreparedChild>,
    root: Box<dyn Planner>,
    share: f64,
    ordered: bool,
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
                    let part_share = per_row * part_rows as f64;
                    units.push(if ordered {
                        let session = session.clone();
                        WorkUnit::run(part_share, move || {
                            // Another runtime thread runs the part while the worker drives its
                            // other parts, and its reads start right away.
                            session
                                .handle()
                                .spawn(run_part(child, scope, planner))
                                .boxed()
                        })
                    } else {
                        WorkUnit::stream(part_share, move || stream_part(child, scope, planner))
                    });
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
        // An unordered pool's lanes take the child's parts in turn with other children's.
        Ok(vec![WorkUnit::turn(units)])
    }
    .boxed()
}

/// Runs a part on a planning run of its own, awaiting only its reads, and yields each of its
/// batches as it finishes, in whatever order their reads arrive.
fn stream_part(
    child: Arc<PreparedChild>,
    scope: WorkScope,
    planner: Box<dyn Planner>,
) -> UnitStream {
    let mut run = Run::new();
    run.admit(planner, scope, child.io.session());
    stream::unfold(Some((run, child)), |state| async move {
        let (mut run, child) = state?;
        loop {
            let progress = match run.advance() {
                Ok(progress) => progress,
                Err(err) => return Some((Err(err), None)),
            };
            match progress {
                Progress::Batch(batch) => return Some((Ok(batch.array), Some((run, child)))),
                Progress::Spawn(scope, planner) => {
                    run.admit(planner, scope, child.io.session());
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return None,
                Progress::Waiting => {
                    if let Err(err) = future::poll_fn(|cx| run.poll_completion(cx)).await {
                        return Some((Err(err), None));
                    }
                }
            }
        }
    })
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
