// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lowers reconstruction and storage-only reads into the scan plan.

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
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
use vortex_scan::plan::PlanRef;
use vortex_scan::plan::PlanSlots;
use vortex_scan::plan::PlanVTable;
use vortex_scan::plan::optimizer::PlanParentReduceRule;
use vortex_scan::plan::optimizer::PlanParentReduceRuleAdapter;
use vortex_scan::plan::optimizer::PlanParentRuleSet;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::STORAGE_SLOT;
use super::expr::rebind_to_storage;

/// Reconstructs extension values from batches of storage.
#[derive(Clone, Debug)]
pub(super) struct ExtensionPack;

impl ExtensionPack {
    fn ext_dtype(plan: &Plan<Self>) -> &ExtDTypeRef {
        plan.dtype()
            .as_extension_opt()
            .vortex_expect("extension pack inherits the validated extension layout dtype")
    }
}

impl PlanVTable for ExtensionPack {
    type PlanData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.ext_pack");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("ExtensionPack", 1)?;
        let storage = slots.required(STORAGE_SLOT)?;
        vortex_ensure_eq!(storage.dtype(), Self::ext_dtype(plan).storage_dtype());
        vortex_ensure_eq!(storage.row_count(), plan.row_count());

        Ok(())
    }

    fn slot_name(_plan: &Plan<Self>, _slot: usize) -> Cow<'_, str> {
        Cow::Borrowed("storage")
    }

    fn reduce_parent(
        plan: &Plan<Self>,
        parent: &PlanRef,
        slot: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        static RULES: PlanParentRuleSet =
            PlanParentRuleSet::new(&[&PlanParentReduceRuleAdapter::new(StorageEvalRule)]);
        RULES.evaluate(&plan.to_plan(), parent, slot, session)
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let chain = compiler.compile(&plan.slot_required(STORAGE_SLOT)?, rows, mask)?;
        Ok(chain.map(|chain| {
            chain.with(ExtensionPackStage {
                ext_dtype: Self::ext_dtype(plan).clone(),
            })
        }))
    }
}

/// Removes reconstruction when the parent evaluates only the storage or its validity.
#[derive(Debug)]
struct StorageEvalRule;

impl PlanParentReduceRule<ExtensionPack> for StorageEvalRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<ExtensionPack>,
        parent: &EvalPlan,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let Some(expr) = rebind_to_storage(
            parent.expression(),
            ExtensionPack::ext_dtype(child).storage_dtype(),
        )?
        else {
            return Ok(None);
        };

        let storage = child.slot_required(STORAGE_SLOT)?;
        Ok(Some(EvalPlan::try_new(expr, storage)?.into_plan()))
    }
}

struct ExtensionPackStage {
    ext_dtype: ExtDTypeRef,
}

impl Operator for ExtensionPackStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };

        Ok(Step::Final(
            ExtensionArray::try_new(self.ext_dtype.clone(), batch)?.into_array(),
        ))
    }
}
