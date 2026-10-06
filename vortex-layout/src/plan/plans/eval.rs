// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::exec::EvalNode;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Selection;
use crate::plan::exec::fuse_dictionary_predicate;
use crate::plan::optimizer::PlanReduceRule;
use crate::plan::pipeline::GraphBuilder;
use crate::plan::pipeline::ops;

/// Applies an expression to the output of its child.
#[derive(Clone, Debug)]
pub struct Eval;

/// The expression evaluated by an [`Eval`].
#[derive(Clone, Debug)]
pub struct EvalData {
    expression: BoundExpression,
    encoded_predicate: bool,
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
            data: EvalData {
                encoded_predicate: matches!(
                    expression.as_opt::<Binary>(),
                    Some(Operator::And | Operator::Or)
                ) && is_infallible(&expression),
                expression,
            },
        }
        .into_typed()
    }

    /// Returns the expression evaluated by this plan.
    pub fn expression(&self) -> &BoundExpression {
        &self.data().expression
    }

    /// Applies the whole predicate to encoded values before expanding row results. Layout
    /// planning cannot push into encodings discovered only when a segment is decoded.
    pub(crate) fn apply(&self, mut array: ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
        if !self.data().encoded_predicate {
            return array.apply_bound(self.expression());
        }
        if let Some(dict) = array.as_opt::<Dict>()
            && !dict.codes().dtype().is_nullable()
            && dict.values().len() <= dict.codes().len()
        {
            let values = dict.values().clone().apply_bound(self.expression())?;
            return Ok(DictArray::try_new(dict.codes().clone(), values)?.into_array());
        }
        let mut ctx = ExecutionCtx::new(session.clone());
        if array
            .as_opt::<Slice>()
            .is_some_and(|slice| slice.child().is::<RunEnd>() || slice.child().is::<Pco>())
        {
            array = array.execute::<ArrayRef>(&mut ctx)?;
        }
        if let Some(runend) = array.as_opt::<RunEnd>() {
            // Each comparison otherwise decompresses the values and expands its own boolean
            // runs. Share the values across the compound predicate and expand only its result.
            let values = SharedArray::new(runend.values().clone())
                .into_array()
                .apply_bound(self.expression())?;
            return Ok(RunEnd::try_new_offset_length(
                runend.ends().clone(),
                values,
                runend.offset(),
                array.len(),
                &mut ctx,
            )?
            .into_array());
        }
        if array.is::<Pco>() {
            // PCO has no comparison kernel; each branch would decompress the same values.
            array = SharedArray::new(array).into_array();
        }
        let result = array.apply_bound(self.expression())?;
        fuse_dictionary_predicate(result, &mut ctx)
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

    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        Ok(Box::new(EvalNode::new(
            plan.clone(),
            Selection::try_new(rows, mask)?,
            ctx.session().clone(),
        )))
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        cx: &mut GraphBuilder<'_>,
    ) -> VortexResult<()> {
        ops::eval(plan, rows, mask, cx)
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

fn is_infallible(expression: &BoundExpression) -> bool {
    expression
        .as_scalar()
        .is_none_or(|scalar| scalar.signature().is_infallible())
        && expression.children().iter().all(is_infallible)
}
