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

    let mut children: Option<Vec<PlanRef>> = None;
    for (index, child) in plan.children().iter_refs().enumerate() {
        let child = child?;
        let optimized = optimize(child.clone())?;
        if let Some(children) = &mut children {
            children.push(optimized);
        } else if !PlanRef::ptr_eq(child, &optimized) {
            let mut optimized_children = Vec::with_capacity(plan.child_count());
            for previous in plan.children().iter_refs().take(index) {
                optimized_children.push(previous?.clone());
            }
            optimized_children.push(optimized);
            children = Some(optimized_children);
        }
    }

    let Some(children) = children else {
        return Ok(plan);
    };

    let plan = plan.with_children(children)?;
    if let Some(rewritten) = reduce(&plan)? {
        return optimize(rewritten);
    }
    Ok(plan)
}
