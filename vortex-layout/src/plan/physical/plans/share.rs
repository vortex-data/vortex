// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::arrays::shared::Shared;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::check_child_count;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::operators::UNBOUNDED;
use crate::plan::physical::operators::WholeSource;
use crate::plan::physical::optimize;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

/// Produces its child's whole value once per split, as a
/// [`SharedArray`](crate::arrays::shared::SharedArray) that the split's later scans reuse.
///
/// Dictionary values sit under a share. An expression the optimizer pushes onto the values stays
/// above it, so the expression runs over values canonicalized once, and the projection reading the
/// same dictionary finds them already canonical, rather than each evaluating over the encoded
/// values.
#[derive(Clone, Debug)]
pub struct Share;

/// Operator-specific data for a [`Share`] plan.
#[derive(Clone, Debug, Default)]
pub struct ShareData {
    // Equivalent optimizer rewrites retain the original share's identity. Keeping the plan alive
    // also prevents its address from being reused while a rewritten share still refers to it.
    cache_key: Option<PlanRef>,
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

    fn cache_key(&self) -> PlanRef {
        self.data()
            .cache_key
            .clone()
            .unwrap_or_else(|| self.to_plan())
    }

    /// Optimizes the child while retaining the identity of the equivalent whole value.
    pub(crate) fn optimize_child(&self, session: &VortexSession) -> VortexResult<PlanRef> {
        let child = self.child_plan()?;
        let optimized = optimize(child.clone(), session)?;
        if PlanRef::ptr_eq(&child, &optimized) {
            return Ok(self.to_plan());
        }
        Ok(PlanParts {
            vtable: Share,
            dtype: self.dtype().clone(),
            row_count: self.row_count(),
            children: vec![optimized].into(),
            data: ShareData {
                cache_key: Some(self.cache_key()),
            },
        }
        .into_plan())
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
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_err!("Share child is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_bail!("Share child does not match the share's dtype and row count");
        }
        // Generic replacement can change values even when dtype and row count still match.
        data.cache_key = None;
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }

    /// A share whose value an earlier scan of the split produced emits the selected rows of it;
    /// otherwise it collects its child whole, keeps it for the split, and emits the selected
    /// rows.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        let cache = compiler.values().clone();
        let shared = plan.cache_key();
        if let Some(value) = cache.get(&shared) {
            return Ok(Some(Chain::new(OnceSource::new(selected_rows(
                &value, &rows, mask,
            )?))));
        }
        let child = plan.child_plan()?;
        let len = usize::try_from(plan.row_count())?;
        let Some(chain) = compiler.compile(&child, 0..plan.row_count(), &Mask::new_true(len))?
        else {
            return Ok(None);
        };
        let mask = mask.clone();
        let source = WholeSource::new(plan.dtype().clone(), move |whole, _cx| {
            let value = cache.insert(&shared, Shared::new(whole).into_array());
            selected_rows(&value, &rows, &mask)
        });
        Ok(Some(compiler.join_with(vec![chain], |_| UNBOUNDED, source)))
    }
}

/// The rows of `rows` that `mask` selects, from a whole shared value.
fn selected_rows(value: &ArrayRef, rows: &Range<u64>, mask: &Mask) -> VortexResult<ArrayRef> {
    let array = value.slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?;
    if mask.all_true() {
        return Ok(array);
    }
    array.filter(mask.clone())
}
