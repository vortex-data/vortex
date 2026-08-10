// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::stream::FuturesOrdered;
use vortex_array::Canonical;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::label_bound_tree;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::RowIdx as RowIdxFn;
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
use crate::plan::optimizer::PlanParentReduceRule;

/// Concatenates its children row-wise.
#[derive(Clone, Debug)]
pub struct Concat;

/// Row offsets of each concatenated child.
#[derive(Clone, Debug)]
pub struct ConcatData {
    row_offsets: Arc<[u64]>,
}

/// A plan that concatenates its children row-wise.
pub type ConcatPlan = Plan<Concat>;

impl ConcatPlan {
    /// Creates a concatenation from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// Every child must have `dtype`; `row_offsets` must contain the cumulative row offset of
    /// every child; and the sum of all child row counts must equal `row_count` without overflow.
    pub(crate) unsafe fn from_children_unchecked(
        dtype: DType,
        row_count: u64,
        row_offsets: Arc<[u64]>,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Concat,
            dtype,
            row_count,
            children,
            data: ConcatData { row_offsets },
        }
        .into_typed()
    }

    /// Creates a concatenation over `children`.
    ///
    /// Every child must produce `dtype`, and the row domain is the sum of the child row counts.
    pub fn try_new(dtype: DType, children: Vec<PlanRef>) -> VortexResult<Self> {
        let mut row_offsets = Vec::with_capacity(children.len());
        let mut row_count = 0u64;
        for child in &children {
            if child.dtype() != &dtype {
                vortex_bail!(
                    "Concat child dtype {} does not match {dtype}",
                    child.dtype()
                );
            }
            row_offsets.push(row_count);
            row_count = row_count
                .checked_add(child.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Concat row count overflow"))?;
        }
        // SAFETY: Child dtypes and the checked cumulative row metadata were validated above.
        Ok(unsafe {
            Self::from_children_unchecked(dtype, row_count, row_offsets.into(), children.into())
        })
    }

    /// Returns the first row of each child within this plan's row domain.
    pub fn row_offsets(&self) -> &[u64] {
        &self.data().row_offsets
    }
}

impl PlanVTable for Concat {
    type PlanData = ConcatData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.concat");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Row offsets are derived from the children, so nothing needs storing.
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if children.len() != plan.children().len() {
            vortex_bail!(
                "Concat expects {} children but got {}",
                plan.children().len(),
                children.len()
            );
        }

        let mut row_offsets = Vec::with_capacity(children.len());
        let mut row_count = 0u64;
        for child in children.iter() {
            let child = child?;
            if child.dtype() != plan.dtype() {
                vortex_bail!(
                    "Concat child dtype {} does not match {}",
                    child.dtype(),
                    plan.dtype()
                );
            }
            row_offsets.push(row_count);
            row_count = row_count
                .checked_add(child.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Concat row count overflow"))?;
        }
        if row_count != plan.row_count() {
            vortex_bail!(
                "Concat children have {row_count} rows but the plan has {}",
                plan.row_count()
            );
        }
        data.row_offsets = row_offsets.into();
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= plan.row_count(),
            "Concat row range {:?} is outside 0..{}",
            row_range,
            plan.row_count()
        );
        vortex_ensure!(
            mask.len() == usize::try_from(row_range.end - row_range.start)?,
            "Concat mask length mismatch"
        );
        if row_range.is_empty() {
            let empty = Canonical::empty(plan.dtype()).into_array();
            return Ok(future::ready(Ok(empty)).boxed());
        }

        let mut chunk_futures = Vec::new();
        for (chunk, &chunk_offset) in plan.children().iter().zip(plan.row_offsets()) {
            let chunk = chunk?;
            let chunk_end = chunk_offset
                .checked_add(chunk.row_count())
                .ok_or_else(|| vortex_err!("Chunk row offset overflow"))?;
            let start = row_range.start.max(chunk_offset);
            let end = row_range.end.min(chunk_end);
            if start < end {
                let child_range = start - chunk_offset..end - chunk_offset;
                let mask_range = usize::try_from(start - row_range.start)?
                    ..usize::try_from(end - row_range.start)?;
                chunk_futures.push(chunk.execute(ctx, &child_range, mask.slice(mask_range))?);
            }
        }

        Ok(async move {
            let chunks: Vec<_> = FuturesOrdered::from_iter(chunk_futures)
                .try_collect()
                .await?;
            vortex_ensure!(!chunks.is_empty(), "Non-empty row range selected no chunks");
            if chunks.len() == 1 {
                return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
            }
            let dtype = chunks[0].dtype().clone();
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
        .boxed())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("chunks[{index}]"))
    }
}

/// Pushes an expression into every chunk of a [`Concat`].
#[derive(Debug)]
pub(crate) struct ExpressionConcatRule;

impl PlanParentReduceRule<Concat> for ExpressionConcatRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Concat>,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        // Row-index expressions are relative to the whole row domain, so they cannot be evaluated
        // chunk by chunk.
        let references_row_idx = label_bound_tree(
            expression,
            |node| {
                node.as_scalar()
                    .is_some_and(|scalar_fn| scalar_fn.is::<RowIdxFn>())
            },
            |acc, &child| acc | child,
        )
        .get(&ExactBoundExpr(expression.clone()))
        .copied()
        .unwrap_or(false);
        if references_row_idx {
            return Ok(None);
        }

        let chunks = child
            .children()
            .iter()
            .map(|chunk| Ok(EvalPlan::try_new(expression.clone(), chunk?)?.into_plan()))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(
            ConcatPlan::try_new(expression.dtype().clone(), chunks)?.into_plan(),
        ))
    }
}
