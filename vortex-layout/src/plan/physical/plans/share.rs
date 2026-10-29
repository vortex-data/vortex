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
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

/// Shares its child's whole encoded value between consumers, wrapping each consumer's value in a
/// [`SharedArray`](crate::arrays::shared::SharedArray).
///
/// Dictionary values sit under a share. An expression the optimizer pushes onto the values stays
/// above it, so it can run over canonical values. The cache retains the encoded input; a predicate
/// need not keep its much larger decoded input after producing a boolean result. A projecting take
/// caches its shared value separately, so its later batches can reuse the decoded dictionary.
#[derive(Clone, Debug)]
pub struct Share;

/// Operator-specific data for a [`Share`] plan.
#[derive(Clone, Debug, Default)]
pub struct ShareData;

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
            data: ShareData,
        }
        .into_typed()
    }

    /// Returns the shared plan.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
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
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Share", children, 1)?;
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

    /// Reuses a projecting take's shared value when available; otherwise wraps the cached encoded
    /// input, collecting it whole on the first request.
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
        let shared = plan.clone().into_plan();
        if let Some(value) = cache.get(&shared) {
            return Ok(Some(Chain::new(OnceSource::new(selected_rows(
                &value, &rows, mask,
            )?))));
        }
        let child = plan.child_plan()?;
        if let Some(value) = cache.get(&child) {
            return Ok(Some(Chain::new(OnceSource::new(selected_rows(
                &Shared::new(value).into_array(),
                &rows,
                mask,
            )?))));
        }
        let len = usize::try_from(plan.row_count())?;
        let Some(chain) = compiler.compile(&child, 0..plan.row_count(), &Mask::new_true(len))?
        else {
            return Ok(None);
        };
        let mask = mask.clone();
        let source = WholeSource::new(plan.dtype().clone(), move |whole, _cx| {
            let value = Shared::new(cache.insert(&child, whole)).into_array();
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
