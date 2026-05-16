//! Sink that counts distinct non-null `i64` values.
//!
//! Lane-safe: declares `Parallelism::Lanes { max: None }` and keeps
//! a per-lane `HashSet<i64>` in `LocalState`. The hot path
//! (canonicalize + hash) runs without taking any cross-lane lock.
//! When a lane finishes, it drains its local set into the shared
//! `CountDistinctState::set` under a single mutex critical section.
//!
//! `count-distinct` is lane-safe in the general case because the
//! aggregation function (`set union`) is associative and
//! commutative; per-lane partial state can be merged in any order.
//! Vortex's `AggregateFn` will eventually expose this as a property
//! so the engine can pick this strategy automatically.
//!
//! A type-generic version (parameterised on Vortex's primitive
//! trait) is a follow-up; today this operator is i64-only because
//! that's what ClickBench Q5 needs.

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
use vortex_utils::aliases::hash_set::HashSet;

use crate::Cardinality;
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

/// Shared accumulator for `CountDistinctI64Sink`. Each lane drains
/// its private set into this `Mutex<HashSet>` exactly once on
/// finish; multiple sink instances (e.g. one per shard) can share
/// the same `CountDistinctState` to compute a global distinct count.
pub struct CountDistinctState {
    set: Mutex<HashSet<i64>>,
}

impl Default for CountDistinctState {
    fn default() -> Self {
        Self::new()
    }
}

impl CountDistinctState {
    pub fn new() -> Self {
        Self {
            set: Mutex::new(HashSet::default()),
        }
    }

    pub fn distinct_count(&self) -> u64 {
        self.set.lock().len() as u64
    }

    fn merge_lane(&self, mut lane: HashSet<i64>) {
        let mut shared = self.set.lock();
        if shared.is_empty() {
            *shared = lane;
        } else {
            shared.reserve(lane.len());
            shared.extend(lane.drain());
        }
    }
}

/// Per-lane accumulator. Each lane owns its own `HashSet<i64>` and
/// runs the inner ingest loop without contending on the shared
/// mutex. On finish, the lane drains into `CountDistinctState`.
pub struct CountDistinctLocalState {
    set: HashSet<i64>,
    drained: bool,
}

pub struct CountDistinctI64Sink {
    label: String,
    domain: Domain,
    state: Arc<CountDistinctState>,
}

impl CountDistinctI64Sink {
    pub fn new(
        label: impl Into<String>,
        domain: Domain,
        state: Arc<CountDistinctState>,
    ) -> Self {
        Self {
            label: label.into(),
            domain,
            state,
        }
    }
}

impl Operator for CountDistinctI64Sink {
    type GlobalState = ();
    type LocalState = CountDistinctLocalState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.domain.clone(), 1)],None,
        )
        .lanes(None)
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(CountDistinctLocalState {
            set: HashSet::default(),
            drained: false,
        })
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
            // Skip placeholder batches entirely — no work for
            // don't-care rows.
            if !batch.demand_all_false() {
                let demand = batch.demand().clone();
                ingest_i64_array(&batch.into_array(), &demand, &mut local.set)?;
            }
            return Ok(WorkStatus::Made);
        };
        if ctx.input_finished(InputPortId::from_index(0)) {
            if !local.drained {
                local.drained = true;
                let lane_set = std::mem::take(&mut local.set);
                self.state.merge_lane(lane_set);
            }
            return Ok(WorkStatus::Finished);
        };
        Ok(WorkStatus::Made)
    }
}

/// Decode any `i64?` array into a `PrimitiveArray<i64>` and insert
/// non-null demand-true values into the per-lane set. Rows where
/// `demand[i]` is false are placeholder/garbage and are skipped.
fn ingest_i64_array(
    array: &ArrayRef,
    demand: &vortex_mask::Mask,
    set: &mut HashSet<i64>,
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
            .map_err(|_| EngineError::message("count_distinct expected primitive array"))?,
    };
    if primitive.ptype() != PType::I64 {
        return Err(EngineError::message(format!(
            "count_distinct expected i64, got {:?}",
            primitive.ptype()
        )));
    };
    let buffer = primitive.to_buffer::<i64>();
    let validity: Validity = primitive
        .validity()
        .map_err(|e| EngineError::message(format!("validity: {e}")))?;
    set.reserve(buffer.len());
    let demand_all_true = demand.all_true();
    match validity {
        Validity::NonNullable | Validity::AllValid => {
            if demand_all_true {
                for i in 0..buffer.len() {
                    if let Some(v) = buffer.get(i).copied() {
                        set.insert(v);
                    }
                }
            } else {
                for i in 0..buffer.len() {
                    if !demand.value(i) {
                        continue;
                    }
                    if let Some(v) = buffer.get(i).copied() {
                        set.insert(v);
                    }
                }
            }
        }
        Validity::AllInvalid => {}
        Validity::Array(_) => {
            for i in 0..buffer.len() {
                if !demand_all_true && !demand.value(i) {
                    continue;
                }
                let valid = validity.is_valid(i).unwrap_or(false);
                if valid && let Some(v) = buffer.get(i).copied() {
                    set.insert(v);
                }
            }
        }
    };
    Ok(())
}
