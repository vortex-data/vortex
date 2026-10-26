// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::scalar_fn::builtin::pack::Pack;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_expr::BoundExpression;
use vortex_expr::traversal::NodeExt;
use vortex_expr::traversal::Transformed;
use vortex_expr::traversal::TraversalOrder;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::physical::PackPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::check_child_count;
use crate::plan::physical::operators::EvalStage;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::optimizer::PlanReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

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

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let chain = compiler.compile(&plan.child_plan()?, rows, mask)?;
        Ok(chain.map(|chain| chain.with(EvalStage::new(plan.expression().clone()))))
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

/// Eval(expr1, Eval(expr2)) -> Eval(expr1 . expr2)
#[derive(Debug)]
pub(crate) struct EvalEvalRule;
impl PlanParentReduceRule<Eval> for EvalEvalRule {
    type Parent = Eval;
    fn reduce_parent(
        &self,
        child: &Plan<Eval>,
        parent: &Plan<Eval>,
        _: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let child_expr = child.expression();
        let expr = parent.expression().clone().transform_down(|expr| {
            if !expr.is_root() {
                return Ok(Transformed::no(expr));
            }
            Ok(Transformed {
                value: child_expr.clone(),
                order: TraversalOrder::Skip,
                changed: true,
            })
        })?;
        // We want to fold repeated expressions like ($.a).b -> $.a.b.
        let expr = expr.into_inner().optimize()?;
        let plan = EvalPlan::try_new(expr, child.child_plan()?)?;
        Ok(Some(plan.into_plan()))
    }
}

/// Eval(pack({0: expr})) -> PackPlan(expr) for a non-nullable pack
#[derive(Debug)]
pub(crate) struct EvalPackRule;
impl PlanReduceRule<Eval> for EvalPackRule {
    fn reduce(&self, plan: &Plan<Eval>) -> VortexResult<Option<PlanRef>> {
        let Some(expr) = plan.expression.as_opt::<Pack>() else {
            return Ok(None);
        };
        if expr.nullability != Nullability::NonNullable {
            return Ok(None);
        }
        let [child_expr] = plan.expression.children() else {
            return Ok(None);
        };
        let fields = StructFields::new(expr.names.clone(), vec![child_expr.dtype().clone()]);

        let child_plan = plan.child(0)?.vortex_expect("no child plan");
        let child_plan = EvalPlan::try_new(child_expr.clone(), child_plan)?.into_plan();

        let plan = PackPlan::try_new(
            fields,
            Nullability::NonNullable,
            plan.row_count(),
            vec![child_plan],
            None,
        )?;
        Ok(Some(plan.into_plan()))
    }
}
