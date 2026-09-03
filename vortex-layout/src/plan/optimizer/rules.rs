// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and type-erased interfaces for plan rewrites.

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::plan::Plan;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;

/// A rewrite over one concrete plan operator.
///
/// Rules return one rewrite without recursively optimizing the replacement. The plan optimizer
/// owns traversal and drives further rewrites.
pub trait PlanReduceRule<P: PlanVTable>: Debug + Send + Sync + 'static {
    /// Attempts to replace `plan`.
    fn reduce(&self, plan: &Plan<P>) -> VortexResult<Option<PlanRef>>;
}

/// Type-erased interface used by [`PlanRuleSet`].
pub trait DynPlanReduceRule: Debug + Send + Sync + 'static {
    /// Returns whether this rule supports the concrete plan operator.
    fn matches(&self, plan: &PlanRef) -> bool;

    /// Attempts to replace `plan`.
    fn reduce(&self, plan: &PlanRef) -> VortexResult<Option<PlanRef>>;
}

/// Bridges a typed [`PlanReduceRule`] to a type-erased static registry.
pub struct PlanReduceRuleAdapter<P, R> {
    rule: R,
    _plan: PhantomData<fn() -> P>,
}

impl<P, R> PlanReduceRuleAdapter<P, R> {
    /// Creates an adapter for a typed plan rule.
    pub const fn new(rule: R) -> Self {
        Self {
            rule,
            _plan: PhantomData,
        }
    }
}

impl<P, R> Debug for PlanReduceRuleAdapter<P, R>
where
    P: PlanVTable,
    R: PlanReduceRule<P>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PlanReduceRuleAdapter")
            .field("plan", &type_name::<P>())
            .field("rule", &self.rule)
            .finish()
    }
}

impl<P, R> DynPlanReduceRule for PlanReduceRuleAdapter<P, R>
where
    P: PlanVTable,
    R: PlanReduceRule<P>,
{
    fn matches(&self, plan: &PlanRef) -> bool {
        plan.is::<P>()
    }

    fn reduce(&self, plan: &PlanRef) -> VortexResult<Option<PlanRef>> {
        let Some(plan) = plan.as_opt::<P>() else {
            return Ok(None);
        };
        self.rule.reduce(plan)
    }
}

/// An ordered static collection of single-plan rewrite rules.
pub struct PlanRuleSet {
    rules: &'static [&'static dyn DynPlanReduceRule],
}

impl PlanRuleSet {
    /// Creates a rule set whose first successful rewrite wins.
    pub const fn new(rules: &'static [&'static dyn DynPlanReduceRule]) -> Self {
        Self { rules }
    }

    /// Evaluates rules registered for the concrete plan operator.
    pub fn evaluate(&self, plan: &PlanRef) -> VortexResult<Option<PlanRef>> {
        for rule in self.rules {
            if !rule.matches(plan) {
                continue;
            }
            let Some(reduced) = rule.reduce(plan)? else {
                continue;
            };

            #[cfg(debug_assertions)]
            {
                vortex_error::vortex_ensure!(
                    reduced.row_count() == plan.row_count(),
                    "Plan rewrite from {rule:?} changed row count from {} to {}",
                    plan.row_count(),
                    reduced.row_count()
                );
                vortex_error::vortex_ensure!(
                    reduced.dtype() == plan.dtype(),
                    "Plan rewrite from {rule:?} changed dtype from {} to {}",
                    plan.dtype(),
                    reduced.dtype()
                );
            }

            return Ok(Some(reduced));
        }
        Ok(None)
    }
}

/// A metadata-only rewrite where a child plan rewrites its parent plan.
///
/// Rules return one rewrite without recursively optimizing the replacement. The plan optimizer
/// owns traversal and drives further rewrites.
pub trait PlanParentReduceRule<C: PlanVTable>: Debug + Send + Sync + 'static {
    /// The concrete parent operator matched by this rule.
    type Parent: PlanVTable;

    /// Attempts to replace `parent` based on its child at `child_idx`.
    fn reduce_parent(
        &self,
        child: &Plan<C>,
        parent: &Plan<Self::Parent>,
        child_idx: usize,
    ) -> VortexResult<Option<PlanRef>>;
}

/// Type-erased interface used by [`PlanParentRuleSet`].
pub trait DynPlanParentReduceRule: Debug + Send + Sync + 'static {
    /// Returns whether this rule supports the concrete child and parent operators.
    fn matches(&self, child: &PlanRef, parent: &PlanRef) -> bool;

    /// Attempts to replace `parent` based on `child` at `child_idx`.
    fn reduce_parent(
        &self,
        child: &PlanRef,
        parent: &PlanRef,
        child_idx: usize,
    ) -> VortexResult<Option<PlanRef>>;
}

/// Bridges a typed [`PlanParentReduceRule`] to a type-erased static registry.
pub struct PlanParentReduceRuleAdapter<C, R> {
    rule: R,
    _child: PhantomData<fn() -> C>,
}

impl<C, R> PlanParentReduceRuleAdapter<C, R> {
    /// Creates an adapter for a typed parent-child rule.
    pub const fn new(rule: R) -> Self {
        Self {
            rule,
            _child: PhantomData,
        }
    }
}

impl<C, R> Debug for PlanParentReduceRuleAdapter<C, R>
where
    C: PlanVTable,
    R: PlanParentReduceRule<C>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PlanParentReduceRuleAdapter")
            .field("parent", &type_name::<R::Parent>())
            .field("child", &type_name::<C>())
            .field("rule", &self.rule)
            .finish()
    }
}

impl<C, R> DynPlanParentReduceRule for PlanParentReduceRuleAdapter<C, R>
where
    C: PlanVTable,
    R: PlanParentReduceRule<C>,
{
    fn matches(&self, child: &PlanRef, parent: &PlanRef) -> bool {
        child.is::<C>() && parent.is::<R::Parent>()
    }

    fn reduce_parent(
        &self,
        child: &PlanRef,
        parent: &PlanRef,
        child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let Some(child) = child.as_opt::<C>() else {
            return Ok(None);
        };
        let Some(parent) = parent.as_opt::<R::Parent>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(child, parent, child_idx)
    }
}

/// An ordered static collection of parent-child plan rewrite rules.
pub struct PlanParentRuleSet {
    rules: &'static [&'static dyn DynPlanParentReduceRule],
}

impl PlanParentRuleSet {
    /// Creates a rule set whose first successful rewrite wins.
    pub const fn new(rules: &'static [&'static dyn DynPlanParentReduceRule]) -> Self {
        Self { rules }
    }

    /// Evaluates rules registered for the concrete `(parent, child)` pair.
    pub fn evaluate(
        &self,
        child: &PlanRef,
        parent: &PlanRef,
        child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        for rule in self.rules {
            if !rule.matches(child, parent) {
                continue;
            }
            let Some(reduced) = rule.reduce_parent(child, parent, child_idx)? else {
                continue;
            };

            #[cfg(debug_assertions)]
            {
                vortex_error::vortex_ensure!(
                    reduced.row_count() == parent.row_count(),
                    "Plan rewrite from {rule:?} changed row count from {} to {}",
                    parent.row_count(),
                    reduced.row_count()
                );
                vortex_error::vortex_ensure!(
                    reduced.dtype() == parent.dtype(),
                    "Plan rewrite from {rule:?} changed dtype from {} to {}",
                    parent.dtype(),
                    reduced.dtype()
                );
            }

            return Ok(Some(reduced));
        }
        Ok(None)
    }
}
