// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! The optimizer applies static rewrites top-down, optimizes children, then retries rewrites
//! exposed by the optimized children.

use vortex_error::VortexResult;

use crate::plan::PlanRef;
use crate::plan::optimizer::reduce_parent;
use crate::plan::optimizer::reduce_plan;

fn reduce(plan: &PlanRef) -> VortexResult<Option<PlanRef>> {
    if let Some(rewritten) = reduce_plan(plan)? {
        return Ok(Some(rewritten));
    }
    for child_idx in 0..plan.child_count() {
        if let Some(rewritten) = reduce_parent(plan, child_idx)? {
            return Ok(Some(rewritten));
        }
    }
    Ok(None)
}

/// Optimizes `plan`, preserving its dtype and row domain.
pub fn optimize(plan: PlanRef) -> VortexResult<PlanRef> {
    if let Some(rewritten) = reduce(&plan)? {
        return optimize(rewritten);
    }

    let mut children = Vec::with_capacity(plan.child_count());
    let mut changed = false;
    for child in plan.children().iter() {
        let child = child?;
        let optimized = optimize(child.clone())?;
        changed |= !PlanRef::ptr_eq(&child, &optimized);
        children.push(optimized);
    }

    if !changed {
        return Ok(plan);
    }

    let plan = plan.with_children(children)?;
    if let Some(rewritten) = reduce(&plan)? {
        return optimize(rewritten);
    }
    Ok(plan)
}
