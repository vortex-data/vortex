// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! The optimizer applies static rewrites top-down, optimizes children, then retries rewrites
//! exposed by the optimized children.

use rustc_hash::FxHashMap;
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
///
/// A plan shared by several parents, as the source of a query is by its conjuncts and its
/// projection, is optimized once.
pub fn optimize(plan: PlanRef) -> VortexResult<PlanRef> {
    optimize_shared(plan, &mut FxHashMap::default())
}

/// The optimized form of each plan optimized so far, by the address of its shared inner.
type Memo = FxHashMap<usize, PlanRef>;

fn optimize_shared(plan: PlanRef, memo: &mut Memo) -> VortexResult<PlanRef> {
    let key = plan.as_ptr_key();
    if let Some(optimized) = memo.get(&key) {
        return Ok(optimized.clone());
    }
    let optimized = optimize_once(plan, memo)?;
    memo.insert(key, optimized.clone());
    Ok(optimized)
}

fn optimize_once(plan: PlanRef, memo: &mut Memo) -> VortexResult<PlanRef> {
    if let Some(rewritten) = reduce(&plan)? {
        return optimize_shared(rewritten, memo);
    }

    let mut children = Vec::with_capacity(plan.child_count());
    let mut changed = false;
    for child in plan.children().iter() {
        let child = child?;
        let optimized = optimize_shared(child.clone(), memo)?;
        changed |= !PlanRef::ptr_eq(&child, &optimized);
        children.push(optimized);
    }

    if !changed {
        return Ok(plan);
    }

    let plan = plan.with_children(children)?;
    if let Some(rewritten) = reduce(&plan)? {
        return optimize_shared(rewritten, memo);
    }
    Ok(plan)
}
