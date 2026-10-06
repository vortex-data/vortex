// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

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
use crate::plan::exec::ShareNode;
use crate::plan::pipeline::GraphBuilder;
use crate::plan::pipeline::ops;

/// Produces its child's whole value once, as a [`SharedArray`](vortex_array::arrays::SharedArray)
/// that every execution of the plan reuses.
///
/// Dictionary values sit under a share. An expression the optimizer pushes onto the values stays
/// above it, so the expression runs over values canonicalized once, and the projection reading the
/// same dictionary finds them already canonical, rather than each evaluating over the encoded
/// values.
#[derive(Clone, Debug)]
pub struct Share;

/// Operator-specific data for a [`Share`] plan: the shared value, once an execution produced it.
///
/// A plan rebuilt with new children starts empty.
#[derive(Clone, Default)]
pub struct ShareData {
    value: Arc<Mutex<Option<ArrayRef>>>,
}

impl fmt::Debug for ShareData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShareData")
            .field("cached", &self.value.lock().is_some())
            .finish()
    }
}

/// A plan that shares its child's value between executions.
pub type SharePlan = Plan<Share>;

impl SharePlan {
    /// Creates a share of `child`.
    pub fn new(child: PlanRef) -> Self {
        PlanParts {
            vtable: Share,
            dtype: child.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child].into(),
            data: ShareData::default(),
        }
        .into_typed()
    }

    /// Returns the shared plan.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
    }

    /// The value an earlier execution produced, if any.
    pub(crate) fn cached(&self) -> Option<ArrayRef> {
        self.data().value.lock().clone()
    }

    /// Records the value for later executions, keeping the first when two race.
    pub(crate) fn cache(&self, value: ArrayRef) -> ArrayRef {
        self.data().value.lock().get_or_insert(value).clone()
    }
}

impl PlanVTable for Share {
    type PlanData = ShareData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.share");
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
        check_child_count("Share", children, 1)?;
        *data = ShareData::default();
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_err!("Share child is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_bail!("Share child does not match the share's dtype and row count");
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
        _ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        Ok(Box::new(ShareNode::new(
            plan.clone(),
            Selection::try_new(rows, mask)?,
        )))
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        cx: &mut GraphBuilder<'_>,
    ) -> VortexResult<()> {
        ops::share(plan, rows, mask, cx)
    }
}
