//! Sink that consumes `i64?` array batches and appends every
//! non-null value into a shared `Vec<i64>`.
//!
//! Used by point-lookup queries (e.g. Q20) where the upstream filter
//! has already applied the predicate, so the sink just records what
//! arrived. A more general sink would parameterise on the value type
//! via Vortex's typed primitive trait — that's a follow-up.

use std::sync::Arc;
use std::task::Context;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineError;
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

pub struct CollectI64Sink {
    label: String,
    domain: Domain,
    captured: Arc<Mutex<Vec<i64>>>,
}

impl CollectI64Sink {
    pub fn new(
        label: impl Into<String>,
        domain: Domain,
        captured: Arc<Mutex<Vec<i64>>>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            captured,
        }
    }
}

impl Operator for CollectI64Sink {
    type GlobalState = ();
    type LocalState = bool;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],
            None,
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
        // Filtered output has unknown cardinality; we just want
        // every row that arrives.
        inputs[0].require_span(DomainSpan::new(0, u64::MAX / 2));
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
            if !batch.demand_all_false() {
                let demand = batch.demand().clone();
                ingest_i64_into_vec(&batch.into_array(), &demand, &self.captured)?;
            }
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            *local = true;
            return Ok(WorkStatus::Finished);
        };
        Ok(WorkStatus::Made)
    }
}

/// Append non-null demand-true `i64` values into `sink`. Rows with
/// `demand[i] == false` are placeholder/garbage and skipped.
fn ingest_i64_into_vec(
    array: &ArrayRef,
    demand: &vortex_mask::Mask,
    sink: &Arc<Mutex<Vec<i64>>>,
) -> EngineResult<()> {
    #[expect(deprecated)]
    let canonical = array
        .to_canonical()
        .map_err(|e| EngineError::message(format!("canonicalize: {e}")))?;
    let primitive: PrimitiveArray = match canonical {
        Canonical::Primitive(p) => p,
        other => other
            .into_array()
            .try_downcast::<Primitive>()
            .map_err(|_| EngineError::message("collect_i64 expected primitive array"))?,
    };
    if primitive.ptype() != PType::I64 {
        return Err(EngineError::message(format!(
            "collect_i64 expected i64, got {:?}",
            primitive.ptype()
        )));
    };
    let buffer = primitive.to_buffer::<i64>();
    let validity: Validity = primitive
        .validity()
        .map_err(|e| EngineError::message(format!("validity: {e}")))?;
    let mut sink = sink.lock();
    let demand_all_true = demand.all_true();
    for i in 0..buffer.len() {
        if !demand_all_true && !demand.value(i) {
            continue;
        }
        let valid = match &validity {
            Validity::NonNullable | Validity::AllValid => true,
            Validity::AllInvalid => false,
            Validity::Array(_) => validity.is_valid(i).unwrap_or(false),
        };
        if valid && let Some(v) = buffer.get(i).copied() {
            sink.push(v);
        }
    };
    Ok(())
}
