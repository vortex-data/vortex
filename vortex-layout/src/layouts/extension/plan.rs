// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scan plan of an extension layout.

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;
use vortex_scan::execution::operator::Cx;
use vortex_scan::execution::operator::Input;
use vortex_scan::execution::operator::Operator;
use vortex_scan::execution::operator::Step;
use vortex_scan::execution::pipeline::Chain;
use vortex_scan::execution::pipeline::Compiler;
use vortex_scan::plan::Eval;
use vortex_scan::plan::EvalPlan;
use vortex_scan::plan::Plan;
use vortex_scan::plan::PlanId;
use vortex_scan::plan::PlanParts;
use vortex_scan::plan::PlanRef;
use vortex_scan::plan::PlanSlots;
use vortex_scan::plan::PlanVTable;
use vortex_scan::plan::optimizer::PlanParentReduceRule;
use vortex_scan::plan::optimizer::PlanParentReduceRuleAdapter;
use vortex_scan::plan::optimizer::PlanParentRuleSet;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::extension::expr::rebind_to_storage;

const STORAGE: usize = 0;

/// Assembles extension values from the storage in its only slot.
///
/// An [`Eval`] above it whose expression reads the extension only through `ext_storage(root)`,
/// `is_null(root)` or `is_not_null(root)` is rewritten to evaluate over the storage instead, so the
/// storage plan can push the expression further down. Compiled, it wraps each storage batch into
/// extension values.
#[derive(Clone, Debug)]
pub struct ExtPack;

/// Operator-specific extension assembly data.
#[derive(Clone, Debug)]
pub struct ExtPackData;

/// A plan that assembles extension values from their storage.
pub type ExtPackPlan = Plan<ExtPack>;

impl ExtPack {
    /// Creates an extension assembly from potentially unbuilt slots without validation.
    ///
    /// # Safety
    ///
    /// `slots` must hold exactly one present slot producing the storage dtype of `ext_dtype` over
    /// `row_count` rows.
    pub(crate) unsafe fn from_slots_unchecked(
        ext_dtype: ExtDTypeRef,
        row_count: u64,
        slots: PlanSlots,
    ) -> ExtPackPlan {
        PlanParts {
            vtable: ExtPack,
            dtype: DType::Extension(ext_dtype),
            row_count,
            slots,
            data: ExtPackData,
        }
        .into_typed()
    }

    /// Creates an assembly of `ext_dtype` values from a plan producing their storage.
    ///
    /// # Errors
    ///
    /// Returns an error if `storage` does not produce the storage dtype of `ext_dtype`.
    pub fn try_new_plan(ext_dtype: ExtDTypeRef, storage: PlanRef) -> VortexResult<ExtPackPlan> {
        let row_count = storage.row_count();
        let slots = [storage].into();
        validate_slots(&ext_dtype, row_count, &slots)?;

        // SAFETY: The slot shape was validated above.
        Ok(unsafe { Self::from_slots_unchecked(ext_dtype, row_count, slots) })
    }

    /// Returns the extension dtype `plan` assembles.
    pub fn ext_dtype(plan: &ExtPackPlan) -> &ExtDTypeRef {
        plan.dtype()
            .as_extension_opt()
            .vortex_expect("ExtPack dtype must be an extension")
    }

    /// Returns the plan producing the storage of `plan`.
    pub fn storage(plan: &ExtPackPlan) -> VortexResult<PlanRef> {
        plan.slot_required(STORAGE)
    }
}

impl PlanVTable for ExtPack {
    type PlanData = ExtPackData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.ext_pack");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The extension dtype is the plan dtype.
        Some(EmptyMetadata)
    }

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        validate_slots(ExtPack::ext_dtype(plan), plan.row_count(), slots)
    }

    fn slot_name(_plan: &Plan<Self>, slot: usize) -> Cow<'_, str> {
        match slot {
            STORAGE => Cow::Borrowed("storage"),
            _ => Cow::Owned(format!("slot[{slot}]")),
        }
    }

    fn reduce_parent(
        plan: &Plan<Self>,
        parent: &PlanRef,
        slot: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        static RULES: PlanParentRuleSet =
            PlanParentRuleSet::new(&[&PlanParentReduceRuleAdapter::new(ExpressionExtPackRule)]);
        RULES.evaluate(&plan.to_plan(), parent, slot, session)
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let chain = compiler.compile(&ExtPack::storage(plan)?, rows, mask)?;
        let ext_dtype = ExtPack::ext_dtype(plan).clone();
        Ok(chain.map(|chain| chain.with(ExtWrapStage::new(ext_dtype))))
    }
}

fn validate_slots(ext_dtype: &ExtDTypeRef, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    slots.check_len("ExtPack", STORAGE + 1)?;
    let storage = slots.required(STORAGE)?;
    vortex_ensure_eq!(storage.dtype(), ext_dtype.storage_dtype());
    vortex_ensure_eq!(storage.row_count(), row_count);
    Ok(())
}

/// Rewrites an [`Eval`] over an [`ExtPack`] into an [`Eval`] over its storage, when the expression
/// reads only the storage.
#[derive(Debug)]
struct ExpressionExtPackRule;

impl PlanParentReduceRule<ExtPack> for ExpressionExtPackRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &ExtPackPlan,
        parent: &Plan<Eval>,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let storage = ExtPack::storage(child)?;
        let Some(expr) = rebind_to_storage(parent.expression(), storage.dtype())? else {
            return Ok(None);
        };
        Ok(Some(EvalPlan::try_new(expr, storage)?.into_plan()))
    }
}

/// Wraps each storage batch into extension values.
struct ExtWrapStage {
    ext_dtype: ExtDTypeRef,
}

impl ExtWrapStage {
    fn new(ext_dtype: ExtDTypeRef) -> Self {
        Self { ext_dtype }
    }
}

impl Operator for ExtWrapStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        Ok(Step::Final(
            ExtensionArray::try_new(self.ext_dtype.clone(), batch)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_scan::plan::PlanRef;
    use vortex_session::registry::ReadContext;

    use super::ExtPack;
    use crate::layouts::flat::FlatLayout;
    use crate::segments::SegmentId;
    use crate::test::TestPoint;

    fn scan(dtype: DType) -> VortexResult<PlanRef> {
        FlatLayout::new(4, dtype, SegmentId::from(0), ReadContext::new([])).new_plan()
    }

    #[test]
    fn try_new_plan_requires_the_storage_dtype() -> VortexResult<()> {
        let ext_dtype = TestPoint::dtype(Nullability::Nullable);

        let plan =
            ExtPack::try_new_plan(ext_dtype.clone(), scan(ext_dtype.storage_dtype().clone())?)?;
        assert_eq!(plan.dtype(), &DType::Extension(ext_dtype.clone()));
        assert_eq!(plan.row_count(), 4);

        let storage = DType::Primitive(PType::I32, Nullability::Nullable);
        assert!(ExtPack::try_new_plan(ext_dtype, scan(storage)?).is_err());
        Ok(())
    }
}
