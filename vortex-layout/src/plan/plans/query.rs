// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::QueryNode;
use crate::plan::exec::Selection;
use crate::plan::optimize;
use crate::scan::filter::FilterExpr;

/// A filter and a projection over one source, evaluated the way a scan evaluates them.
///
/// The filter is split into conjuncts, each planned over the source on its own. An execution
/// evaluates them one at a time, in the order the shared [`FilterExpr`] prefers, each under the
/// rows the earlier ones kept, so a selective conjunct spares the later ones most of the rows.
/// The projection, planned over the source, then runs under the rows that passed every conjunct.
/// A split whose rows no conjunct keeps produces nothing and reads nothing more.
///
/// The conjunct order adapts across executions: every execution reports each conjunct's
/// selectivity to the shared scheduler, as the splits of a scan do.
#[derive(Clone, Debug)]
pub struct Query;

/// The scheduler of a [`Query`]'s conjuncts.
#[derive(Clone)]
pub struct QueryData {
    scheduler: Option<Arc<FilterExpr>>,
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
        Ok(PlanParts {
            vtable: Query,
            dtype: projection.dtype().clone(),
            row_count: projection.row_count(),
            children: children.into(),
            data: QueryData {
                scheduler: scheduler.map(Arc::new),
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

    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        Ok(Box::new(QueryNode::new(
            plan.clone(),
            Selection::try_new(rows, mask)?,
            ctx.session().clone(),
        )))
    }
}
