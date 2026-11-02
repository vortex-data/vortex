// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::Canonical;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_array::scalar_fn::is_negative_cost;
use vortex_error::VortexResult;
use vortex_expr::BoundLabels;
use vortex_expr::label_bound_tree;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::plan::physical::Eval;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::Share;
use crate::plan::physical::check_child_count;
use crate::plan::physical::operators::EvalStage;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::operators::TakeSource;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

const CODES: usize = 0;
const VALUES: usize = 1;

/// Indexes `values` by `codes`, with children ordered as `[codes, values]`.
#[derive(Clone, Debug)]
pub struct Take;

/// A plan that indexes one child by another.
pub type TakePlan = Plan<Take>;

impl TakePlan {
    /// Creates a take from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// `children` must be `[codes, values]`; `codes` must have `row_count` rows; and `dtype` must
    /// be the values dtype unioned with the codes nullability.
    pub(crate) unsafe fn from_children_unchecked(
        dtype: DType,
        row_count: u64,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Take,
            dtype,
            row_count,
            children,
            data: (),
        }
        .into_typed()
    }

    /// Creates a take of `values` at `codes`.
    ///
    /// The row domain is that of `codes`, and the output dtype is that of `values`.
    pub fn new(codes: PlanRef, values: PlanRef) -> Self {
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        let row_count = codes.row_count();
        PlanParts {
            vtable: Take,
            dtype,
            row_count,
            children: vec![codes, values].into(),
            data: (),
        }
        .into_typed()
    }

    /// Returns the plan producing indices.
    pub fn codes(&self) -> VortexResult<PlanRef> {
        self.child_required(CODES)
    }

    /// Returns the plan producing the values being indexed.
    pub fn values(&self) -> VortexResult<PlanRef> {
        self.child_required(VALUES)
    }
}

impl PlanVTable for Take {
    type PlanData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.take");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Take", children, 2)?;
        let codes = children
            .get(CODES)?
            .ok_or_else(|| vortex_error::vortex_err!("Take codes child is absent"))?;
        let values = children
            .get(VALUES)?
            .ok_or_else(|| vortex_error::vortex_err!("Take values child is absent"))?;
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        if codes.row_count() != plan.row_count() || &dtype != plan.dtype() {
            vortex_error::vortex_bail!("Take child shape does not match the plan output");
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            CODES => Cow::Borrowed("codes"),
            VALUES => Cow::Borrowed("values"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    /// A take whose values an earlier scan of the split produced looks its codes up in them;
    /// otherwise it joins its codes with the whole of its values.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let cache = compiler.values().clone();
        let values_plan = plan.values()?;
        let cached_values = cache.get(&values_plan);
        if cached_values.is_none()
            && (mask.true_count() as u64) < values_plan.row_count()
            && let Some((selected, expression)) = predicate_over_selected_codes(plan, &values_plan)?
        {
            return Ok(compiler
                .compile_rows(&selected.into_plan(), rows, mask)?
                .map(|chain| chain.with(EvalStage::new(expression))));
        }
        let Some(codes) = compiler.compile_rows(&plan.codes()?, rows, mask)? else {
            return Ok(None);
        };
        if let Some(values) = cached_values {
            let source = TakeSource::new(plan.clone(), Some(values), None);
            return Ok(Some(compiler.join(vec![codes], source)));
        }
        let values = values_plan;
        let values_rows = 0..values.row_count();
        let len = usize::try_from(values_rows.end)?;
        let values = match compiler.compile(&values, values_rows, &Mask::new_true(len))? {
            Some(values) => values,
            None => Chain::new(OnceSource::new(
                Canonical::empty(values.dtype()).into_array(),
            )),
        };
        let source = TakeSource::new(plan.clone(), None, Some(cache));
        Ok(Some(compiler.join_with(
            vec![codes, values],
            TakeSource::capacity,
            source,
        )))
    }
}

/// Pushes an expression, or the part of it that reads the values, onto the dictionary values of
/// a [`Take`].
///
/// Evaluating over values rather than codes is only sound for infallible expressions, since values
/// no selected row references are evaluated too, and, when codes may be null, strict ones:
/// otherwise per-row behaviour is not preserved.
///
/// A boolean expression is pushed whole onto the values as they are, so it stays above any
/// [`Share`] of them. Otherwise the largest part holding every read of the values and built only
/// from negative-cost functions, such as the byte length in `cast(byte_length($))`, is pushed
/// beneath the share, since it is cheaper over the encoded values than canonicalizing them; the
/// rest stays above the take.
#[derive(Debug)]
pub(crate) struct ExpressionTakeRule;

/// Per expression node: whether it reads the root, and whether it is strict, infallible, and built
/// only from negative-cost functions.
type Labels = BoundLabels<(bool, bool, bool, bool)>;

fn expression_labels(expression: &BoundExpression, codes_nullable: bool) -> Labels {
    label_bound_tree(
        expression,
        |node| match node.as_scalar() {
            Some(scalar_fn) => (
                false,
                scalar_fn.signature().is_strict() || !codes_nullable,
                scalar_fn.signature().is_infallible(),
                is_negative_cost(scalar_fn.id()),
            ),
            None => (true, true, true, true),
        },
        |acc, &child| {
            (
                acc.0 | child.0,
                acc.1 & child.1,
                acc.2 & child.2,
                acc.3 & child.3,
            )
        },
    )
}

/// Evaluates a pushed predicate after selecting its codes when that avoids evaluating more
/// dictionary entries than selected rows. Taking the original values also shares them with a
/// later projection, while negative-cost expressions keep their access to encoded values.
fn predicate_over_selected_codes(
    plan: &TakePlan,
    values: &PlanRef,
) -> VortexResult<Option<(TakePlan, BoundExpression)>> {
    let Some(eval) = values.as_opt::<Eval>() else {
        return Ok(None);
    };
    if !eval.dtype().is_boolean() || eval.expression().contains::<DynamicComparison>()? {
        return Ok(None);
    }
    let codes = plan.codes()?;
    let labels = expression_labels(eval.expression(), codes.dtype().is_nullable());
    if !matches!(
        labels.get(&ExactBoundExpr(eval.expression().clone())),
        Some((true, true, true, false))
    ) {
        return Ok(None);
    }
    let input = eval.child_plan()?;
    let selected = TakePlan::new(codes, input.clone());
    let Some(expression) = rebind_root(eval.expression(), input.dtype(), selected.dtype()) else {
        return Ok(None);
    };
    if expression.dtype() != plan.dtype() {
        return Ok(None);
    }
    Ok(Some((selected, expression)))
}

impl PlanParentReduceRule<Take> for ExpressionTakeRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Take>,
        parent: &Plan<Eval>,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        // A dynamic comparison stays above the take: evaluated per batch it reads the latest
        // bound, and the dictionary batches the take emits evaluate it over their values anyway.
        // Values it never reaches can always be kept between scans.
        if expression.contains::<DynamicComparison>()? {
            return Ok(None);
        }
        // Strictness keeps a null code null. Without null codes every row takes its value's result.
        let codes_nullable = child.codes()?.dtype().is_nullable();
        let labels = expression_labels(expression, codes_nullable);
        let label = |node: &BoundExpression| {
            labels
                .get(&ExactBoundExpr(node.clone()))
                .copied()
                .unwrap_or((false, false, false, false))
        };

        let (references_root, is_strict, is_infallible, is_negative_cost) = label(expression);
        if references_root && is_strict && is_infallible && !is_negative_cost {
            if !expression.dtype().is_boolean() {
                return Ok(None);
            }
            let values = child.values()?;
            let Some(expression) = rebind_root(expression, child.dtype(), values.dtype()) else {
                return Ok(None);
            };
            let values = EvalPlan::try_new(expression, values)?.into_plan();
            let take = TakePlan::new(child.codes()?, values).into_plan();
            if take.dtype() != parent.dtype() {
                return Ok(None);
            }
            return Ok(Some(take));
        }

        let Some(inner) = negative_cost_part(expression, &labels) else {
            return Ok(None);
        };
        let values = child.values()?;
        let values = match values.as_opt::<Share>() {
            Some(share) => share.child_plan()?,
            None => values,
        };
        let Some(pushed) = rebind_root(&inner, child.dtype(), values.dtype()) else {
            return Ok(None);
        };
        let values = EvalPlan::try_new(pushed, values)?.into_plan();
        let take = TakePlan::new(child.codes()?, values).into_plan();
        let outer = replace(
            expression,
            &inner,
            &BoundExpression::new_root(take.dtype().clone()),
        )?;
        if outer.dtype() != expression.dtype() {
            return Ok(None);
        }
        if outer.is_root() {
            return Ok(Some(take));
        }
        Ok(Some(EvalPlan::try_new(outer, take)?.into_plan()))
    }
}

/// The largest strict, infallible, negative-cost part of `expression` holding every read of the
/// root, unless that is the bare root.
fn negative_cost_part(expression: &BoundExpression, labels: &Labels) -> Option<BoundExpression> {
    let (references_root, is_strict, is_infallible, is_negative_cost) =
        labels.get(&ExactBoundExpr(expression.clone())).copied()?;
    if !references_root {
        return None;
    }
    if is_strict && is_infallible && is_negative_cost {
        return (!expression.is_root()).then(|| expression.clone());
    }
    let mut reading = expression.children().iter().filter(|child| {
        labels
            .get(&ExactBoundExpr((*child).clone()))
            .is_some_and(|label| label.0)
    });
    let child = reading.next()?;
    if reading.next().is_some() {
        return None;
    }
    negative_cost_part(child, labels)
}

/// `expression`, bound to the take's dtype `from`, bound instead to its values' dtype `to`, or
/// `None` when it does not apply there.
///
/// Values are less nullable than the take when its codes may be null: the take's nulls come from
/// its codes, which a strict expression keeps null.
fn rebind_root(expression: &BoundExpression, from: &DType, to: &DType) -> Option<BoundExpression> {
    if from == to {
        return Some(expression.clone());
    }
    replace(
        expression,
        &BoundExpression::new_root(from.clone()),
        &BoundExpression::new_root(to.clone()),
    )
    .ok()
}

/// `expression` with each occurrence of `needle` replaced by `replacement`.
fn replace(
    expression: &BoundExpression,
    needle: &BoundExpression,
    replacement: &BoundExpression,
) -> VortexResult<BoundExpression> {
    if ExactBoundExpr(expression.clone()) == ExactBoundExpr(needle.clone()) {
        return Ok(replacement.clone());
    }
    if expression.children().is_empty() {
        return Ok(expression.clone());
    }
    let children = expression
        .children()
        .iter()
        .map(|child| replace(child, needle, replacement))
        .collect::<VortexResult<Vec<_>>>()?;
    expression.clone().with_children(children)
}
