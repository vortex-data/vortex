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
    // Concat has no local rewrites. Optimizing its independent chunks eagerly traverses the
    // entire file even when a partition only scans a small row range.
    if let Some(concat) = plan.as_opt::<Concat>() {
        let session = session.clone();
        return Ok(concat
            .map_chunks(plan.dtype().clone(), move |chunk| optimize(chunk, &session))
            .into_plan());
    }
    if let Some(rewritten) = reduce(&plan, session)? {
        return optimize(rewritten, session);
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
            let optimized = optimize(child.clone(), session)?;
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
    if let Some(rewritten) = reduce(&plan, session)? {
        return optimize(rewritten, session);
    }
    Ok(plan)
}
