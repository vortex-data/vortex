// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The parts of a prepared scan, the units a host's threads take, and the planner of each.
//!
//! A scan's parts are cut by [`split_points`](crate::split::split_points), which both engines
//! share, over the chunk boundaries [`chunk_boundaries`] finds in the scan's plans.

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use crate::plan::PlanRef;
use crate::plan::builtins::Concat;
use crate::planning::planner::Planner;

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

/// The planner of one part of `scan`.
///
/// With a filter, the part first evaluates every pruning plan over its own rows, on the thread
/// that takes it, from the zone maps the file's cache holds. A static conjunct is so applied once
/// to each row, by the part that holds it. A dynamic conjunct, such as the bound a top-N pushes
/// down, is applied as it stands when the part is taken, so later parts skip more as it tightens.
pub(crate) fn part_planner(scan: &Arc<PreparedScan>, part: Range<u64>) -> Box<dyn Planner> {
    let chunks = {
        let scan = Arc::clone(scan);
        let part = part.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, part, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::for_part(
            Arc::clone(scan),
            part,
            Box::new(chunks),
        ))
    } else {
        chunks(None)
    }
}
