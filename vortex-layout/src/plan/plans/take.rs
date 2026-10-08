// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_error::VortexExpect;
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
use crate::plan::check_child_count;
use crate::plan::exec::ExecContext;
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
/// depend only on the plan. Every execution of the plan, across the splits of a scan, shares one
/// copy: later executions skip reading and decoding them, and see the same array. A plan rebuilt
/// with new children starts empty.
#[derive(Clone)]
pub struct TakeData {
    values: Arc<OnceLock<ArrayRef>>,
    /// Whether the values may be kept. Values evaluated with a dynamic comparison change as the
    /// engine updates it, so each execution evaluates them again.
    cacheable: bool,
}

impl Default for TakeData {
    fn default() -> Self {
        Self {
            values: Default::default(),
            cacheable: true,
        }
    }
}

impl TakeData {
    fn for_values(values: &PlanRef) -> VortexResult<Self> {
        let dynamic = match values.as_opt::<Eval>() {
            Some(eval) => eval.expression().contains::<DynamicComparison>()?,
            None => false,
        };
        Ok(Self {
            values: Default::default(),
            cacheable: !dynamic,
        })
    }
}

impl fmt::Debug for TakeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TakeData")
            .field("values_cached", &self.values.get().is_some())
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
        let data = TakeData::for_values(&values).vortex_expect("expression traversal cannot fail");
        PlanParts {
            vtable: Take,
            dtype,
            row_count,
            children: vec![codes, values].into(),
            data,
        }
        .into_typed()
    }

    /// The values an earlier execution of this plan produced, if any.
    pub(crate) fn cached_values(&self) -> Option<ArrayRef> {
        self.data().values.get().cloned()
    }

    /// Records the values for later executions, keeping the first when two race, unless the
    /// values may not be kept.
    pub(crate) fn cache_values(&self, values: ArrayRef) -> ArrayRef {
        if !self.data().cacheable {
            return values;
        }
        self.data().values.get_or_init(|| values).clone()
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
        let codes = children
            .get(CODES)?
            .ok_or_else(|| vortex_error::vortex_err!("Take codes child is absent"))?;
        let values = children
            .get(VALUES)?
            .ok_or_else(|| vortex_error::vortex_err!("Take values child is absent"))?;
        // New children may produce different values.
        *data = TakeData::for_values(&values)?;
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

    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        _ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        Ok(Box::new(TakeNode::new(
            plan.clone(),
            Selection::try_new(rows, mask)?,
        )))
    }
}

/// Pushes a strict, infallible boolean expression onto the dictionary values of a [`Take`].
#[derive(Debug)]
pub(crate) struct ExpressionTakeRule;

impl PlanParentReduceRule<Take> for ExpressionTakeRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Take>,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        if !expression.dtype().is_boolean() {
            return Ok(None);
        }
        // Evaluating over values rather than codes is only sound when the expression reads the
        // root, is strict, and cannot fail: otherwise per-row behaviour is not preserved.
        let labels = label_bound_tree(
            expression,
            |node| match node.as_scalar() {
                Some(scalar_fn) => (
                    false,
                    scalar_fn.signature().is_strict(),
                    scalar_fn.signature().is_infallible(),
                ),
                None => (true, true, true),
            },
            |acc, &child| (acc.0 | child.0, acc.1 & child.1, acc.2 & child.2),
        );
        let (references_root, is_strict, is_infallible) = labels
            .get(&ExactBoundExpr(expression.clone()))
            .copied()
            .unwrap_or((false, false, false));
        if !references_root || !is_strict || !is_infallible {
            return Ok(None);
        }

        let values = EvalPlan::try_new(expression.clone(), child.values()?)?.into_plan();
        Ok(Some(TakePlan::new(child.codes()?, values).into_plan()))
    }
}
