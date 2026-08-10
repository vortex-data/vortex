// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan optimization.
//!
//! Optimization is driven top-down from [`Eval`] nodes, which apply the static parent-reduction
//! rules in [`crate::plan::optimizer`] as they become applicable. Operators without a rule simply
//! optimize their children.

use vortex_error::VortexResult;

use crate::plan::Eval;
use crate::plan::PlanRef;

/// Optimizes `plan`, preserving its dtype and row domain.
pub fn optimize(plan: PlanRef) -> VortexResult<PlanRef> {
    if let Some(eval) = plan.as_opt::<Eval>() {
        return eval.optimize_top_down(None);
    }

    let children = plan
        .children()
        .iter()
        .map(|child| optimize(child?))
        .collect::<VortexResult<Vec<_>>>()?;
    plan.with_children(children)
}
