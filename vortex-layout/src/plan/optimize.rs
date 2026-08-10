// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic bottom-up optimization over physical plans.

use vortex_error::VortexResult;

use crate::plan::Eval;
use crate::plan::PlanRef;

/// Optimizes `plan`, preserving its dtype and row domain.
pub fn optimize(plan: PlanRef) -> VortexResult<PlanRef> {
    let mut children = Vec::with_capacity(plan.child_count());
    let mut changed = false;
    for child in plan.children().iter() {
        let child = child?;
        let optimized = optimize(child.clone())?;
        changed |= !PlanRef::ptr_eq(&child, &optimized);
        children.push(optimized);
    }

    let plan = if changed {
        plan.with_children(children)?
    } else {
        plan
    };

    let Some(eval) = plan.as_opt::<Eval>() else {
        return Ok(plan);
    };
    if eval.expression().is_root() {
        return eval.child_plan();
    }
    Ok(plan)
}
