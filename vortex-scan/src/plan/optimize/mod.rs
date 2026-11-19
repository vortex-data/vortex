// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! The optimizer applies static rewrites top-down, optimizes the children in every present slot,
//! then retries rewrites exposed by the optimized children.

mod registry;
mod rules;

pub use rules::*;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use self::registry::reduce_parent;
use self::registry::reduce_plan;
use crate::plan::Concat;
use crate::plan::LazyPlanRef;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;

#[derive(Clone, Copy)]
enum OptimizeMode {
    /// Preserves the result of the plan at each row.
    Exact,
    /// Allows weaker falsifiers as long as true still proves that a row can be discarded.
    Pruning,
}

fn reduce(
    plan: &PlanRef,
    session: &VortexSession,
    mode: OptimizeMode,
) -> VortexResult<Option<PlanRef>> {
    if let Some(rewritten) = reduce_plan(plan, session)? {
        return Ok(Some(rewritten));
    }
    for (slot, child) in plan.slots().present() {
        if let Some(rewritten) = reduce_parent(child.get()?, plan, slot, session, mode)? {
            return Ok(Some(rewritten));
        }
    }
    Ok(None)
}

/// Optimizes `plan`, preserving its dtype and row domain.
pub fn optimize(plan: PlanRef, session: &VortexSession) -> VortexResult<PlanRef> {
    optimize_with_mode(plan, session, OptimizeMode::Exact)
}

/// Optimizes a falsifier whose true rows can be discarded without reading the data.
pub(crate) fn optimize_pruning(plan: PlanRef, session: &VortexSession) -> VortexResult<PlanRef> {
    optimize_with_mode(plan, session, OptimizeMode::Pruning)
}

fn optimize_with_mode(
    plan: PlanRef,
    session: &VortexSession,
    mode: OptimizeMode,
) -> VortexResult<PlanRef> {
    // Concat has no local rewrites. Optimizing its independent chunks eagerly traverses the
    // entire file even when a partition only scans a small row range.
    if let Some(concat) = plan.as_opt::<Concat>() {
        let session = session.clone();
        return Ok(concat
            .map_chunks(plan.dtype().clone(), move |chunk| {
                optimize_with_mode(chunk, &session, mode)
            })
            .into_plan());
    }
    if let Some(rewritten) = reduce(&plan, session, mode)? {
        return optimize_with_mode(rewritten, session, mode);
    }

    // Absent slots stay absent, and unchanged slots keep their shared lazy slot.
    let mut changed = false;
    let slots = plan
        .slots()
        .iter()
        .map(|slot| {
            let Some(lazy) = slot else {
                return Ok(None);
            };
            let child = lazy.get()?;
            let optimized = optimize_with_mode(child.clone(), session, mode)?;
            if PlanRef::ptr_eq(child, &optimized) {
                return Ok(Some(lazy.clone()));
            }
            changed = true;
            Ok(Some(LazyPlanRef::ready(optimized)))
        })
        .collect::<VortexResult<PlanSlots>>()?;
    if !changed {
        return Ok(plan);
    }

    let plan = plan.with_slots(slots)?;
    if let Some(rewritten) = reduce(&plan, session, mode)? {
        return optimize_with_mode(rewritten, session, mode);
    }
    Ok(plan)
}
