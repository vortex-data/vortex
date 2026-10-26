// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! The optimizer applies static rewrites top-down, optimizes children, then retries rewrites
//! exposed by the optimized children.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::plan::physical::Concat;
use crate::plan::physical::PlanRef;
use crate::plan::physical::optimizer::reduce_parent;
use crate::plan::physical::optimizer::reduce_plan;

fn reduce(plan: &PlanRef, session: &VortexSession) -> VortexResult<Option<PlanRef>> {
    if let Some(rewritten) = reduce_plan(plan, session)? {
        return Ok(Some(rewritten));
    }
    for child_idx in 0..plan.child_count() {
        if let Some(rewritten) = reduce_parent(plan, child_idx, session)? {
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

    let mut children: Option<Vec<PlanRef>> = None;
    for (index, child) in plan.children().iter_refs().enumerate() {
        let child = child?;
        let optimized = optimize(child.clone(), session)?;
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
    if let Some(rewritten) = reduce(&plan, session)? {
        return optimize(rewritten, session);
    }
    Ok(plan)
}
