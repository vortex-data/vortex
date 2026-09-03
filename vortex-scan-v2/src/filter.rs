// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexResult;
use vortex_layout::plan::PlanExecutionContext;
use vortex_layout::plan::PlanRef;
use vortex_layout::scan::FilterExpr;
use vortex_mask::Mask;

/// Controls how a scan executes top-level filter conjunctions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FilterMode {
    /// Optimizes the complete predicate as one plan, allowing independent branches to run in
    /// parallel.
    #[default]
    Parallel,
    /// Splits top-level conjunctions and executes them as an adaptively ordered mask chain.
    Adaptive,
}

#[derive(Clone)]
pub(crate) enum FilterPlan {
    Parallel(PlanRef),
    Adaptive {
        filter: Arc<FilterExpr>,
        plans: Arc<[PlanRef]>,
    },
}

impl FilterPlan {
    pub(crate) fn parallel(plan: PlanRef) -> Self {
        Self::Parallel(plan)
    }

    pub(crate) fn adaptive(filter: FilterExpr, plans: Vec<PlanRef>) -> Self {
        Self::Adaptive {
            filter: Arc::new(filter),
            plans: plans.into(),
        }
    }

    pub(crate) fn plans(&self) -> Vec<&PlanRef> {
        match self {
            Self::Parallel(plan) => vec![plan],
            Self::Adaptive { plans, .. } => plans.iter().collect(),
        }
    }

    pub(crate) fn execute(
        &self,
        execution: &PlanExecutionContext,
        row_range: &Range<u64>,
        row_mask: Mask,
    ) -> VortexResult<MaskFuture> {
        match self {
            Self::Parallel(filter) => {
                let predicate =
                    filter.execute(execution, row_range, MaskFuture::ready(row_mask.clone()))?;
                let session = execution.session().clone();
                Ok(MaskFuture::new(row_mask.len(), async move {
                    let predicate = predicate.await?;
                    let mut execution = session.create_execution_ctx();
                    let predicate: Mask = predicate.null_as_false().execute(&mut execution)?;
                    Ok(row_mask.intersect_by_rank(&predicate))
                }))
            }
            Self::Adaptive { filter, plans } => {
                let execution = execution.clone();
                let row_range = row_range.clone();
                let filter = Arc::clone(filter);
                let plans = Arc::clone(plans);
                Ok(MaskFuture::new(row_mask.len(), async move {
                    let mut row_mask = row_mask;
                    let mut remaining = BitVec::from_elem(plans.len(), true);
                    while let Some(index) = filter.next_conjunct(&remaining) {
                        remaining.set(index, false);
                        if row_mask.all_false() {
                            break;
                        }

                        let input_rows = row_mask.true_count();
                        let predicate = plans[index].execute(
                            &execution,
                            &row_range,
                            MaskFuture::ready(row_mask.clone()),
                        )?;
                        let predicate = predicate.await?;
                        let mut ctx = execution.session().create_execution_ctx();
                        let predicate: Mask = predicate.null_as_false().execute(&mut ctx)?;
                        row_mask = row_mask.intersect_by_rank(&predicate);
                        filter.report_selectivity(index, row_mask.density());
                        tracing::trace!(
                            target: "vortex_scan_v2::execution",
                            conjunct = index,
                            input_rows,
                            output_rows = row_mask.true_count(),
                            "applied an adaptive plan filter conjunct"
                        );
                    }
                    Ok(row_mask)
                }))
            }
        }
    }
}
