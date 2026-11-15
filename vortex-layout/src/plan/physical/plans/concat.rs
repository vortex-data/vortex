// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::plan::physical::Eval;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::LazyPlanRef;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::operators::ConcatSource;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;
use crate::plan::physical::pipeline::overlapping;
use crate::plan::physical::selection::selected_slice;

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
    /// Creates a concatenation from potentially unbuilt slots, one per chunk, without
    /// validation.
    ///
    /// # Safety
    ///
    /// Every slot must be present with `dtype`; `row_offsets` must contain the cumulative row
    /// offset of every chunk; and the sum of all chunk row counts must equal `row_count` without
    /// overflow.
    pub(crate) unsafe fn from_slots_unchecked(
        dtype: DType,
        row_count: u64,
        row_offsets: Arc<[u64]>,
        slots: PlanSlots,
    ) -> Self {
        PlanParts {
            vtable: Concat,
            dtype,
            row_count,
            slots,
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
            Self::from_slots_unchecked(dtype, row_count, row_offsets.into(), children.into())
        })
    }

    /// Returns the first row of each child within this plan's row domain.
    pub fn row_offsets(&self) -> &[u64] {
        &self.data().row_offsets
    }

    /// Defers independent child rewrites until a scan reaches the child. Row offsets remain
    /// valid because each materialized result must preserve its child's row count.
    pub(crate) fn map_chunks(
        &self,
        dtype: DType,
        map: impl Fn(PlanRef) -> VortexResult<PlanRef> + Send + Sync + 'static,
    ) -> Self {
        let map = Arc::new(map);
        let slots = self
            .slots()
            .iter()
            .map(|slot| {
                let chunk = slot.clone()?;
                let map = Arc::clone(&map);
                let dtype = dtype.clone();
                Some(LazyPlanRef::lazy(move || {
                    let chunk = chunk.get()?;
                    let mapped = map(chunk.clone())?;
                    vortex_ensure_eq!(mapped.dtype(), &dtype);
                    vortex_ensure_eq!(mapped.row_count(), chunk.row_count());
                    Ok(mapped)
                }))
            })
            .collect();
        PlanParts {
            vtable: Concat,
            dtype,
            row_count: self.row_count(),
            data: self.data().clone(),
            slots,
        }
        .into_typed()
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

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("Concat", plan.slots().len())?;

        let mut row_offsets = Vec::with_capacity(slots.len());
        let mut row_count = 0u64;
        for index in 0..slots.len() {
            let child = slots.required(index)?;
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

    fn slot_name(_plan: &Plan<Self>, slot: usize) -> Cow<'_, str> {
        Cow::Owned(format!("chunks[{slot}]"))
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let capacity = if mask.all_true() {
            overlapping(plan, &rows).size_hint().1.unwrap_or(0)
        } else {
            0
        };
        let mut chains = Vec::with_capacity(capacity);
        for (index, start, local) in overlapping(plan, &rows) {
            let Some((span, local_mask)) = selected_slice(
                mask,
                usize::try_from(local.start - rows.start)?
                    ..usize::try_from(local.end - rows.start)?,
            ) else {
                continue;
            };
            let chunk = plan.slot_required(index)?;
            if let Some(chain) = compiler.compile(
                &chunk,
                rows.start + span.start as u64 - start..rows.start + span.end as u64 - start,
                &local_mask,
            )? {
                chains.push(chain);
            }
        }
        Ok(match chains.len() {
            0 => None,
            // Rows inside one chunk are that chunk's rows: no concatenation is built.
            1 => chains.pop(),
            count => Some(compiler.join(chains, ConcatSource::new(count))),
        })
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
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression().clone();
        Ok(Some(
            child
                .map_chunks(expression.dtype().clone(), move |chunk| {
                    Ok(EvalPlan::try_new(expression.clone(), chunk)?.into_plan())
                })
                .into_plan(),
        ))
    }
}
