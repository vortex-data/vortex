// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pattern-matching for layout plan nodes, used by
//! [`crate::v2::plans::pushdown::PlanPushdownRule`] to identify candidates.
//!
//! Parking-lot module: only `AnyPlan` implements `Matcher`, and
//! nothing calls either today. Lives here so future rule-based
//! pushdown can build matchers without re-introducing the surface.

use crate::v2::plans::LayoutPlanRef;

/// A matcher describes a shape of [`LayoutPlanRef`] that a pushdown
/// rule wants to apply to.
pub trait Matcher {
    type Match<'a>;

    fn matches(plan: &LayoutPlanRef) -> bool {
        Self::try_match(plan).is_some()
    }

    fn try_match(plan: &LayoutPlanRef) -> Option<Self::Match<'_>>;
}

/// Wildcard matcher — matches any plan node.
#[derive(Debug)]
pub struct AnyPlan;

impl Matcher for AnyPlan {
    type Match<'a> = &'a LayoutPlanRef;

    #[inline(always)]
    fn matches(_plan: &LayoutPlanRef) -> bool {
        true
    }

    #[inline(always)]
    fn try_match(plan: &LayoutPlanRef) -> Option<Self::Match<'_>> {
        Some(plan)
    }
}
