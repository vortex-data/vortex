// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! The optimizer applies static rewrites top-down, optimizes the children in every present slot,
//! then retries rewrites exposed by the optimized children.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::plan::physical::Concat;
use crate::plan::physical::LazyPlanRef;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::optimizer::reduce_parent;
use crate::plan::physical::optimizer::reduce_plan;

fn reduce(plan: &PlanRef, session: &VortexSession) -> VortexResult<Option<PlanRef>> {
    if let Some(rewritten) = reduce_plan(plan, session)? {
        return Ok(Some(rewritten));
    }
    for (slot, child) in plan.slots().present() {
        if let Some(rewritten) = reduce_parent(child.get()?, plan, slot, session)? {
            return Ok(Some(rewritten));
        }
    }
    Ok(None)
}

/// Optimizes `plan`, preserving its dtype and row domain.
pub fn optimize(plan: PlanRef, session: &VortexSession) -> VortexResult<PlanRef> {
    Ok(optimize_borrowed(&plan, session)?.unwrap_or(plan))
}

/// Returns a replacement only when the plan changes, without cloning unchanged child plans.
fn optimize_borrowed(plan: &PlanRef, session: &VortexSession) -> VortexResult<Option<PlanRef>> {
    // Concat has no local rewrites. Optimizing its independent chunks eagerly traverses the
    // entire file even when a partition only scans a small row range.
    if let Some(concat) = plan.as_opt::<Concat>() {
        let session = session.clone();
        return Ok(Some(
            concat
                .map_chunks(plan.dtype().clone(), move |chunk| optimize(chunk, &session))
                .into_plan(),
        ));
    }
    if let Some(rewritten) = reduce(plan, session)? {
        return optimize(rewritten, session).map(Some);
    }

    // Rebuilding unchanged slots would clone and drop shared references on every traversal.
    let mut slots: Option<PlanSlots> = None;
    for (index, slot) in plan.slots().iter().enumerate() {
        if let Some(lazy) = slot
            && let Some(optimized) = optimize_borrowed(lazy.get()?, session)?
        {
            slots
                .get_or_insert_with(|| plan.slots().iter().take(index).cloned().collect())
                .push(Some(LazyPlanRef::ready(optimized)));
            continue;
        }
        if let Some(slots) = &mut slots {
            slots.push(slot.clone());
        }
    }
    let Some(slots) = slots else {
        return Ok(None);
    };

    let plan = plan.with_slots(slots)?;
    if let Some(rewritten) = reduce(&plan, session)? {
        return optimize(rewritten, session).map(Some);
    }
    Ok(Some(plan))
}
