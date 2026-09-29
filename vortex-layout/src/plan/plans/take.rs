// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::BoundLabels;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_array::scalar_fn::is_negative_cost;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::Share;
use crate::plan::check_child_count;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Selection;
use crate::plan::exec::TakeNode;
use crate::plan::optimizer::PlanParentReduceRule;

const CODES: usize = 0;
const VALUES: usize = 1;

/// Indexes `values` by `codes`, with children ordered as `[codes, values]`.
#[derive(Clone, Debug)]
pub struct Take;

/// A plan that indexes one child by another.
pub type TakePlan = Plan<Take>;

/// The values of a [`TakePlan`], once any execution of the plan has produced them.
///
/// The values run over their whole domain whatever rows the take is executed with, so they
/// depend only on the plan. Every execution of the plan, across splits and across the filter
/// and projection of a scan, shares one copy: later executions skip reading and decoding them,
/// and see the same array, which lets consumers that cache per dictionary recognise it. A plan
/// rebuilt with new children starts empty.
#[derive(Clone, Default)]
pub struct TakeData {
    values: Arc<Mutex<Option<ArrayRef>>>,
}

impl fmt::Debug for TakeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TakeData")
            .field("values_cached", &self.values.lock().is_some())
            .finish()
    }
}

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
            data: TakeData::default(),
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
        // SAFETY: Parent metadata is derived from the ordered children immediately above.
        unsafe { Self::from_children_unchecked(dtype, row_count, vec![codes, values].into()) }
    }

    /// Returns the plan producing indices.
    pub fn codes(&self) -> VortexResult<PlanRef> {
        self.child_required(CODES)
    }

    /// Returns the plan producing the values being indexed.
    pub fn values(&self) -> VortexResult<PlanRef> {
        self.child_required(VALUES)
    }

    /// The values an earlier execution of this plan produced, if any.
    pub(crate) fn cached_values(&self) -> Option<ArrayRef> {
        self.data().values.lock().clone()
    }

    /// Records the values for later executions, keeping the first when two race.
    pub(crate) fn cache_values(&self, values: ArrayRef) -> ArrayRef {
        self.data().values.lock().get_or_insert(values).clone()
    }
}

impl PlanVTable for Take {
    type PlanData = TakeData;
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
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Take", children, 2)?;
        // New children may produce different values.
        *data = TakeData::default();
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

    fn exec(plan: &Plan<Self>, rows: Range<u64>, mask: Mask) -> VortexResult<Box<dyn ExecNode>> {
        Ok(Box::new(TakeNode::new(
            plan.clone(),
            Selection::try_new(rows, mask)?,
        )))
    }
}

/// Pushes an expression, or the part of it that reads the values, onto the dictionary values of
/// a [`Take`].
///
/// Evaluating over values rather than codes is only sound for strict, infallible expressions:
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

impl PlanParentReduceRule<Take> for ExpressionTakeRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Take>,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        let labels = label_bound_tree(
            expression,
            |node| match node.as_scalar() {
                Some(scalar_fn) => (
                    false,
                    scalar_fn.signature().is_strict(),
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
        );
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
            let values = EvalPlan::try_new(expression.clone(), child.values()?)?.into_plan();
            return Ok(Some(TakePlan::new(child.codes()?, values).into_plan()));
        }

        let Some(inner) = negative_cost_part(expression, &labels) else {
            return Ok(None);
        };
        let values = child.values()?;
        let values = match values.as_opt::<Share>() {
            Some(share) => share.child_plan()?,
            None => values,
        };
        let values = EvalPlan::try_new(inner.clone(), values)?.into_plan();
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
