// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;

use vortex_array::EmptyMetadata;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::optimizer::PlanReduceRule;

/// Applies an expression to the output of its child.
#[derive(Clone, Debug)]
pub struct Eval;

/// The expression evaluated by an [`Eval`].
#[derive(Clone, Debug)]
pub struct EvalData {
    expression: BoundExpression,
}

/// A plan that applies an expression to its child.
pub type EvalPlan = Plan<Eval>;

impl EvalPlan {
    /// Creates an evaluation of `expression`, which must be bound to the child's dtype.
    pub fn try_new(expression: BoundExpression, child: PlanRef) -> VortexResult<Self> {
        validate_expression_child(&expression, &child)?;

        // SAFETY: The expression root dtype was validated against the child dtype above.
        Ok(unsafe { Self::new_unchecked(expression, child) })
    }

    /// Creates an evaluation without validating the expression's root dtype.
    ///
    /// # Safety
    ///
    /// Every scope root in `expression` must have the same dtype as `child`.
    pub unsafe fn new_unchecked(expression: BoundExpression, child: PlanRef) -> Self {
        PlanParts {
            vtable: Eval,
            dtype: expression.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child].into(),
            data: EvalData { expression },
        }
        .into_typed()
    }

    /// Returns the expression evaluated by this plan.
    pub fn expression(&self) -> &BoundExpression {
        &self.data().expression
    }

    /// Returns the child plan supplying the expression root.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
    }
}

impl PlanVTable for Eval {
    type PlanData = EvalData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.eval");
        *ID
    }

    fn fmt(plan: &Plan<Self>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, " expr={}", plan.expression())
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Expressions serialize through `vortex.expr` protobuf, which is not wired up here yet.
        None
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Eval", children, 1)?;
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_error::vortex_err!("Eval child is absent"))?;
        validate_expression_child(plan.expression(), &child)?;
        if child.row_count() != plan.row_count() {
            vortex_error::vortex_bail!(
                "Eval child has {} rows but the plan has {}",
                child.row_count(),
                plan.row_count()
            );
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }
}

fn validate_expression_child(expression: &BoundExpression, child: &PlanRef) -> VortexResult<()> {
    if !expression.is_root_bound_to(child.dtype()) {
        vortex_bail!(
            "Eval expression is not bound to child dtype {}",
            child.dtype()
        );
    }
    Ok(())
}

/// Removes an [`Eval`] whose expression is the identity expression.
#[derive(Debug)]
pub(crate) struct EvalIdentityRule;

impl PlanReduceRule<Eval> for EvalIdentityRule {
    fn reduce(&self, plan: &Plan<Eval>) -> VortexResult<Option<PlanRef>> {
        if plan.expression().is_root() {
            Ok(Some(plan.child_plan()?))
        } else {
            Ok(None)
        }
    }
}
