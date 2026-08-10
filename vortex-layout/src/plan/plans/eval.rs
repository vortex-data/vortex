// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;

use futures::FutureExt;
use vortex_array::EmptyMetadata;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::optimize;
use crate::plan::optimizer::reduce_parent;

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

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        let child = plan.child_plan()?.execute(ctx, row_range, mask)?;
        let expression = plan.expression().clone();
        Ok(async move { child.await?.apply_bound(&expression) }.boxed())
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

impl EvalPlan {
    /// Optimizes this plan top-down, applying parent-reduction rules as they become applicable.
    ///
    /// `blocked_child_type` suppresses one rule re-firing on its own residual output, which would
    /// otherwise loop when a rewrite leaves an expression above the same child kind.
    pub(crate) fn optimize_top_down(
        &self,
        blocked_child_type: Option<PlanId>,
    ) -> VortexResult<PlanRef> {
        if self.expression().is_root() {
            return optimize(self.child_plan()?);
        }

        let child = self.child_plan()?;
        let child_type = child.id();
        let parent = EvalPlan::try_new(self.expression().clone(), child.clone())?.into_plan();
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }

        let child = optimize(child)?;

        let child_type = child.id();
        let parent = EvalPlan::try_new(self.expression().clone(), child)?.into_plan();
        if blocked_child_type != Some(child_type)
            && let Some(rewritten) = reduce_parent(&parent, 0)?
        {
            return Self::optimize_rewrite(rewritten, child_type);
        }
        Ok(parent)
    }

    fn optimize_rewrite(rewritten: PlanRef, previous_child_type: PlanId) -> VortexResult<PlanRef> {
        let Some(eval) = rewritten.as_opt::<Eval>() else {
            return optimize(rewritten);
        };
        // A residual expression may remain above the same child kind after a successful rewrite.
        // Do not immediately apply that rule again; recursively optimize only the retained child.
        let child_type = eval.child_plan()?.id();
        let blocked = (child_type == previous_child_type).then_some(previous_child_type);
        eval.optimize_top_down(blocked)
    }
}

/// Rewrites partition accessors in `expression` to read from a partitioned root.
pub(crate) fn rewrite_partition_root(
    expression: BoundExpression,
    root_dtype: DType,
    collapsed: &[(FieldName, FieldName)],
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if let Some(value_name) = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
            {
                let partition_access = &node.children()[0];
                if let Some(partition_name) = partition_access
                    .as_scalar()
                    .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                    && partition_access.children()[0].is_root()
                    && collapsed.iter().any(|(partition, value)| {
                        partition == partition_name && value == value_name
                    })
                {
                    return Ok(Transformed {
                        value: BoundExpression::try_new(
                            GetItem.bind(partition_name.clone()),
                            [BoundExpression::new_root(root_dtype.clone())],
                        )?,
                        changed: true,
                        order: TraversalOrder::Skip,
                    });
                }
            }

            if node.is_root() {
                Ok(Transformed {
                    value: BoundExpression::new_root(root_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}
