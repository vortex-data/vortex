// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static parent-child rewrite rules for physical plans.

mod rules;

pub use rules::DynPlanParentReduceRule;
pub use rules::PlanParentReduceRule;
pub use rules::PlanParentReduceRuleAdapter;
pub use rules::PlanParentRuleSet;
