// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use futures::FutureExt;
use futures::try_join;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::DictArray;
use vortex_array::dtype::DType;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::optimizer::PlanParentReduceRule;

const CODES: usize = 0;
const VALUES: usize = 1;

/// Indexes `values` by `codes`, with children ordered as `[codes, values]`.
#[derive(Clone, Debug)]
pub struct Take;

/// Whether every dictionary value is referenced by at least one code.
///
/// Lowering carries this over from the dictionary layout, since the operator does not hold one.
#[derive(Clone, Debug)]
pub struct TakeData {
    all_values_referenced: bool,
}

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
        all_values_referenced: bool,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Take,
            dtype,
            row_count,
            children,
            data: TakeData {
                all_values_referenced,
            },
        }
        .into_typed()
    }

    /// Creates a take of `values` at `codes`.
    ///
    /// The row domain is that of `codes`, and the output dtype is that of `values`.
    pub fn new(codes: PlanRef, values: PlanRef) -> Self {
        Self::new_with_all_values_referenced(codes, values, false)
    }

    /// Creates a take that records whether every value is referenced by some code.
    pub fn new_with_all_values_referenced(
        codes: PlanRef,
        values: PlanRef,
        all_values_referenced: bool,
    ) -> Self {
        let dtype = values
            .dtype()
            .union_nullability(codes.dtype().nullability());
        let row_count = codes.row_count();
        // SAFETY: Parent metadata is derived from the ordered children immediately above.
        unsafe {
            Self::from_children_unchecked(
                dtype,
                row_count,
                all_values_referenced,
                vec![codes, values].into(),
            )
        }
    }

    /// Returns whether every value is referenced by at least one code.
    pub fn all_values_referenced(&self) -> bool {
        self.data().all_values_referenced
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

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        let codes_plan = plan.codes()?;
        let values_plan = plan.values()?;
        let codes = codes_plan.execute(ctx, row_range, mask)?;
        let values_len = usize::try_from(values_plan.row_count())?;
        let values = values_plan.execute(
            ctx,
            &(0..values_plan.row_count()),
            MaskFuture::new_true(values_len),
        )?;
        let all_values_referenced = plan.all_values_referenced();

        Ok(async move {
            let (codes, values) = try_join!(codes, values)?;
            // SAFETY: lowering from a dict layout guarantees integer codes and matching dtypes.
            let dictionary = unsafe {
                DictArray::new_unchecked(codes, values)
                    .set_all_values_referenced(all_values_referenced)
            }
            .into_array()
            .optimize()?;
            Ok(dictionary)
        }
        .boxed())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            CODES => Cow::Borrowed("codes"),
            VALUES => Cow::Borrowed("values"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
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
                    scalar_fn.signature().is_fallible(),
                ),
                None => (true, true, false),
            },
            |acc, &child| (acc.0 | child.0, acc.1 & child.1, acc.2 | child.2),
        );
        let (references_root, is_strict, is_fallible) = labels
            .get(&ExactBoundExpr(expression.clone()))
            .copied()
            .unwrap_or((false, false, true));
        if !references_root || !is_strict || is_fallible {
            return Ok(None);
        }

        let values = EvalPlan::try_new(expression.clone(), child.values()?)?.into_plan();
        Ok(Some(TakePlan::new(child.codes()?, values).into_plan()))
    }
}
