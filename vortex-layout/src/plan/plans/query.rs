// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::EmptyMetadata;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::Zoned;
use crate::plan::optimize;
use crate::plan::pipeline::Chain;
use crate::plan::pipeline::Compiler;
use crate::plan::pipeline::Reach;
use crate::plan::pipeline::Shared;
use crate::scan::filter::FilterExpr;

/// A filter and a projection over one source, evaluated the way a scan evaluates them.
///
/// The filter is split into conjuncts, each planned over the source on its own. An execution
/// first prunes with the zone statistics of each conjunct's column, where the conjunct is over a
/// zoned column its zones can prove false, so a split whose zones no conjunct can match reads no
/// data. It then evaluates the conjuncts one at a time, in the order the shared [`FilterExpr`]
/// prefers, each under the rows the earlier ones kept, so a selective conjunct spares the later
/// ones most of the rows. The projection, planned over the source, then runs under the rows that
/// passed every conjunct. A split whose rows nothing keeps produces nothing and reads nothing
/// more.
///
/// The conjunct order adapts across executions: every execution reports each conjunct's
/// selectivity to the shared scheduler, as the splits of a scan do.
#[derive(Clone, Debug)]
pub struct Query;

/// The scheduler of a [`Query`]'s conjuncts, and the pruning plan of each.
#[derive(Clone)]
pub struct QueryData {
    scheduler: Option<Arc<FilterExpr>>,
    /// The selected fraction at or above which a conjunct runs over whole chunks.
    dense_threshold: f64,
    /// Per conjunct, the plan pruning zones for it, once an execution asked.
    pruning: Arc<[OnceLock<Option<PlanRef>>]>,
}

impl fmt::Debug for QueryData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryData")
            .field(
                "conjuncts",
                &self.scheduler.as_ref().map_or(0, |s| s.conjuncts().len()),
            )
            .finish()
    }
}

/// A plan that evaluates a filter and a projection over one source.
pub type QueryPlan = Plan<Query>;

/// The projection's slot in the children. Conjunct `i` is in slot `i + 1`.
const PROJECTION: usize = 0;

/// The selected fraction of a split at or above which a conjunct runs over every row of the
/// chunks holding a selected row and its result is intersected with the mask, rather than
/// running over the selected rows only. The default scan's flat reader uses the same value.
///
/// Filtering an encoded column to a few rows before comparing is cheaper than comparing every
/// row, but the filter has a cost of its own, so a nearly full mask is not worth applying.
pub const DEFAULT_DENSE_THRESHOLD: f64 = 0.2;

impl QueryPlan {
    /// Plans `filter`, if any, and `projection` over `source`.
    ///
    /// Each conjunct of the filter and the projection are pushed into their own copy of
    /// `source` by the optimizer, so each reads only the columns it needs.
    pub fn try_new(
        filter: Option<BoundExpression>,
        projection: BoundExpression,
        source: PlanRef,
    ) -> VortexResult<Self> {
        let scheduler = filter.map(FilterExpr::new);
        let mut children =
            Vec::with_capacity(1 + scheduler.as_ref().map_or(0, |s| s.conjuncts().len()));
        children.push(optimize(
            EvalPlan::try_new(projection, source.clone())?.into_plan(),
        )?);
        if let Some(scheduler) = &scheduler {
            for conjunct in scheduler.conjuncts() {
                vortex_ensure!(
                    conjunct.dtype().is_boolean(),
                    "Query filter conjunct must be boolean, got {}",
                    conjunct.dtype()
                );
                children.push(optimize(
                    EvalPlan::try_new(conjunct.clone(), source.clone())?.into_plan(),
                )?);
            }
        }
        let projection = &children[PROJECTION];
        let children_len = children.len();
        Ok(PlanParts {
            vtable: Query,
            dtype: projection.dtype().clone(),
            row_count: projection.row_count(),
            children: children.into(),
            data: QueryData {
                pruning: (1..children_len).map(|_| OnceLock::new()).collect(),
                scheduler: scheduler.map(Arc::new),
                dense_threshold: DEFAULT_DENSE_THRESHOLD,
            },
        }
        .into_typed())
    }

    /// The plan producing the projected rows.
    pub fn projection(&self) -> VortexResult<PlanRef> {
        self.child_required(PROJECTION)
    }

    /// The plan evaluating conjunct `index`.
    pub fn conjunct(&self, index: usize) -> VortexResult<PlanRef> {
        self.child_required(index + 1)
    }

    /// How many conjuncts the filter has.
    pub fn conjunct_count(&self) -> usize {
        self.children().len() - 1
    }

    /// The shared scheduler of the conjuncts, if there is a filter.
    pub(crate) fn scheduler(&self) -> Option<&Arc<FilterExpr>> {
        self.data().scheduler.as_ref()
    }

    /// The selected fraction at or above which a conjunct runs over whole chunks. See
    /// [`DEFAULT_DENSE_THRESHOLD`].
    pub fn dense_threshold(&self) -> f64 {
        self.data().dense_threshold
    }

    /// This plan with a different [`dense_threshold`](Self::dense_threshold): `0.0` runs every
    /// conjunct over whole chunks, above `1.0` every conjunct over the selected rows only.
    pub fn with_dense_threshold(&self, dense_threshold: f64) -> Self {
        let mut data = self.data().clone();
        data.dense_threshold = dense_threshold;
        PlanParts {
            vtable: Query,
            dtype: self.dtype().clone(),
            row_count: self.row_count(),
            children: self.children().clone(),
            data,
        }
        .into_typed()
    }

    /// The plan telling which rows' zones may hold a row passing conjunct `index`, when the
    /// conjunct is over a zoned column and its zones can prove it false. Built once.
    pub(crate) fn pruning(
        &self,
        index: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let cell = &self.data().pruning[index];
        if let Some(plan) = cell.get() {
            return Ok(plan.clone());
        }
        let plan = match self.conjunct(index)?.as_opt::<Zoned>() {
            Some(zoned) => zoned.pruning_plan(session)?,
            None => None,
        };
        // Another execution may have built it meanwhile; both built the same plan.
        Ok(cell.get_or_init(|| plan).clone())
    }
}

impl PlanVTable for Query {
    type PlanData = QueryData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.query");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The scheduler's conjuncts are not serializable on their own.
        None
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if children.len() != plan.children().len() {
            vortex_bail!(
                "Query expects {} children but got {}",
                plan.children().len(),
                children.len()
            );
        }
        for (index, child) in children.iter().enumerate() {
            let child = child?;
            if child.row_count() != plan.row_count() {
                vortex_bail!("Query child {index} does not cover the query's rows");
            }
            if index == PROJECTION {
                if child.dtype() != plan.dtype() {
                    vortex_bail!("Query projection does not produce the query's dtype");
                }
            } else if !child.dtype().is_boolean() {
                vortex_bail!("Query conjunct {} is not boolean", index - 1);
            }
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == PROJECTION {
            Cow::Borrowed("projection")
        } else {
            Cow::Owned(format!("conjunct[{}]", index - 1))
        }
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let _ = (plan, rows, mask, compiler);
        vortex_bail!("A Query plan runs only at the root of a scan, as stages")
    }

    fn reach(
        plan: &Plan<Self>,
        rows: Range<u64>,
        at: &Reach,
        visit: &mut dyn FnMut(Shared, Range<u64>),
    ) -> VortexResult<()> {
        let _ = (plan, rows, at, visit);
        Ok(())
    }
}
