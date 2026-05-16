//! Sink that captures `ArrayRef`s from incoming batches into a
//! shared `Vec<ArrayRef>`. Useful for tests and the layout-binding
//! convenience runners (`read_full_file`,
//! `read_layout_via_engine`) that want to surface what was
//! materialised without forcing a row-major roundtrip.

use std::sync::Arc;
use std::task::Context;

use parking_lot::Mutex;
use vortex_array::ArrayRef;

use crate::Cardinality;
use crate::Domain;
use crate::DomainSpan;
use crate::EngineResult;
use crate::GlobalInitCtx;
use crate::InputPortId;
use crate::InputPortSpec;
use crate::LocalInitCtx;
use crate::Operator;
use crate::OperatorSpec;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

pub struct ArrayCollectSink {
    label: String,
    domain: Domain,
    captured: Arc<Mutex<Vec<ArrayRef>>>,
}

impl ArrayCollectSink {
    pub fn new(
        label: impl Into<String>,
        domain: Domain,
        captured: Arc<Mutex<Vec<ArrayRef>>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            captured,
        }
    }
}

impl Operator for ArrayCollectSink {
    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],None,
        )
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(false)
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let len = match self.domain.cardinality() {
            Cardinality::Exact(len) => len,
            Cardinality::Unknown => 0,
        };
        inputs[0].require_span(DomainSpan::new(0, len));
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        let port = InputPortId::from_index(0);
        let drainable = ctx.peek(port).is_some() || ctx.input_finished(port);
        let class = if drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::none(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            self.captured.lock().push(batch.into_array());
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            *local = true;
            return Ok(WorkStatus::Finished);
        };
        Ok(WorkStatus::Made)
    }
}
