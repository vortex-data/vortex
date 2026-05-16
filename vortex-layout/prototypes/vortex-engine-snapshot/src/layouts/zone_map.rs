//! Zoned-layout pruning operators.
//!
//! A `vortex.zoned` layout has a *data* child (the actual column
//! data) and an auxiliary *zones* child (a struct of stats per
//! zone — `min`, `max`, `count`, …). At bind time, given an
//! expression that references the data column, we lower it to a
//! pruning predicate over the zones via Vortex's
//! `checked_pruning_expr`. The lowered predicate, evaluated
//! against the zone-stats array, returns one boolean per zone:
//! `true` means the zone *cannot* match — safe to skip.
//!
//! Two operators implement the bind:
//!
//! - [`ZoneMapSink`] is a sink on the zones subgraph. It drains
//!   zone batches, accumulates them into a Vortex `StructArray`,
//!   builds a [`ZoneMap`], evaluates the lowered predicate via
//!   [`ZoneMap::prune`], and publishes the per-zone mask to the
//!   shared [`ZoneMapResource`]. Bumps the resource's version.
//!
//! - [`ZoneMapOperator`] sits between the data subgraph and the
//!   downstream consumer. Input port = data; output port = data.
//!   Holds a clone of the same `Arc<ZoneMapResource>`. On each
//!   data batch it pops, consults the resource for the latest
//!   demand mask over the batch's row range, ANDs with the
//!   batch's own demand, and forwards. When the resource's
//!   `is_range_fully_pruned` returns true for a batch, the
//!   operator emits a placeholder batch instead of waiting for
//!   the data subgraph to do real work.
//!
//! Bind-time wiring (in [`crate::layouts::bind_into_graph`]):
//!
//! ```text
//!         zones subgraph
//!               │
//!               ▼
//!        ┌──────────────┐
//!        │ ZoneMapSink  │ ──── publishes ──►  Arc<ZoneMapResource>
//!        └──────────────┘                            │
//!                                                     │
//!         data subgraph                               │
//!               │                                     │
//!               ▼                                     │
//!        ┌──────────────────┐                         │
//!        │ ZoneMapOperator  │ ◄── reads ──────────────┘
//!        │ (pass-through)   │ ──► output (refined demand)
//!        └──────────────────┘
//! ```

use std::sync::Arc;
use std::task::Context;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;
use vortex_session::VortexSession;

use crate::Batch;
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
use crate::OutputPortSpec;
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
use crate::resources::ZoneMapResource;

// ============================================================================
// ZoneMapSink: zones subgraph → ZoneMapResource
// ============================================================================

pub struct ZoneMapSink {
    label: String,
    domain: Domain,
    column_dtype: DType,
    present_stats: Arc<[Stat]>,
    zone_len: u64,
    data_row_count: u64,
    session: VortexSession,
    resource: Arc<ZoneMapResource>,
}

pub struct ZoneMapSinkState {
    accumulated: Vec<ArrayRef>,
    sealed: bool,
}

impl ZoneMapSink {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        label: impl Into<String>,
        zones_input_domain: Domain,
        column_dtype: DType,
        present_stats: Arc<[Stat]>,
        zone_len: u64,
        data_row_count: u64,
        session: VortexSession,
        resource: Arc<ZoneMapResource>,
    ) -> Self {
        Self {
            label: label.into(),
            domain: zones_input_domain,
            column_dtype,
            present_stats,
            zone_len,
            data_row_count,
            session,
            resource,
        }
    }
}

impl Operator for ZoneMapSink {
    type GlobalState = ();
    type LocalState = ZoneMapSinkState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("zones", self.domain.clone(), 1)],
            // Sink: no output. Side effects via the resource.
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
        Ok(ZoneMapSinkState {
            accumulated: Vec::new(),
            sealed: false,
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
        // We need every zone to compute pruning.
        if let Cardinality::Exact(rows) = self.domain.cardinality()
            && rows > 0
        {
            inputs[0].require_span(DomainSpan::new(0, rows));
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // EV reflects expected pruning savings, not the cost of
        // building the zone map. The cost surfaces upstream as
        // the zones subgraph's own broker proposal — we don't
        // double-count here.
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            WorkClass::Cpu,
            // p_needed_x256 = 128 ≈ 0.5 (half-likely the
            // pruning predicate will save downstream work).
            WorkValue::candidate(self.data_row_count, 128),
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
        if local.sealed {
            return Ok(WorkStatus::Finished);
        }

        // Drain available zone batches.
        while let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            local.accumulated.push(batch.into_array());
        }
        if !ctx.input_finished(InputPortId::from_index(0)) {
            return Ok(WorkStatus::Made);
        }

        // Zones input sealed: assemble the ZoneMap and evaluate
        // the lowered pruning predicate.
        let chunks = std::mem::take(&mut local.accumulated);
        if chunks.is_empty() {
            // No zones: nothing to prune (every zone is "matches"
            // by default). Leave the resource empty; consumers
            // see no pruning.
            local.sealed = true;
            return Ok(WorkStatus::Finished);
        }
        let combined: ArrayRef = if chunks.len() == 1 {
            chunks.into_iter().next().unwrap()
        } else {
            let dtype = chunks[0].dtype().clone();
            ChunkedArray::try_new(chunks, dtype)
                .map_err(|e| EngineError::message(format!("ChunkedArray::try_new: {e}")))?
                .into_array()
        };
        let struct_array: StructArray = combined
            .clone()
            .try_downcast::<Struct>()
            .map_err(|_| EngineError::message("zones array is not a StructArray"))?
            .into();

        let zone_map = ZoneMap::try_new(
            self.column_dtype.clone(),
            struct_array,
            Arc::clone(&self.present_stats),
            self.zone_len,
            self.data_row_count,
        )
        .map_err(|e| EngineError::message(format!("ZoneMap::try_new: {e}")))?;

        // Evaluate the (lowered) pruning predicate against the
        // zone map.
        let predicate = self.resource.pruning_predicate().clone();
        let mask = zone_map
            .prune(&predicate, &self.session)
            .map_err(|e| EngineError::message(format!("ZoneMap::prune: {e}")))?;

        let pruned_zones = mask.true_count();
        ctx.trace(format!(
            "{}: built ZoneMap, pruned {} / {} zones",
            self.label,
            pruned_zones,
            self.resource.zone_count()
        ));

        self.resource.install_zone_map(zone_map);
        self.resource.install_mask(mask);

        local.sealed = true;
        Ok(WorkStatus::Finished)
    }
}

// ============================================================================
// ZoneMapOperator: data subgraph → output, refining demand
// ============================================================================

pub struct ZoneMapOperator {
    label: String,
    input_domain: Domain,
    output_domain: Domain,
    output_columns: usize,
    /// Bind-time row range over the data layout's domain. Spans
    /// emitted upstream are *layout-local* in this range; we
    /// translate to the resource's zone coordinates by adding
    /// `data_row_offset`.
    data_row_offset: u64,
    resource: Arc<ZoneMapResource>,
}

pub struct ZoneMapOperatorState {
    /// Cumulative output rows pushed. Filter-style "consecutive
    /// from 0" emit isn't right here because we preserve the
    /// input domain — instead we forward each batch with its
    /// original span unchanged.
    sealed: bool,
    /// Last `ZoneMapResource::version()` this operator translated
    /// against. `update` polls the resource and only requests a
    /// fresh propagation pass when the version has actually
    /// advanced — without this, a resource that publishes once
    /// and then sits stable would still cause a propagate re-fire
    /// after every `run`, billing back-prop overhead to a query
    /// whose demand isn't actually changing.
    last_seen_resource_version: u64,
}

impl ZoneMapOperator {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        output_domain: Domain,
        output_columns: usize,
        data_row_offset: u64,
        resource: Arc<ZoneMapResource>,
    ) -> Self {
        Self {
            label: label.into(),
            input_domain,
            output_domain,
            output_columns,
            data_row_offset,
            resource,
        }
    }

    /// Return the row at which the zone covering `row` ends.
    /// Used to advance the requirement walk one zone at a time.
    fn zone_end_at(&self, row: u64) -> u64 {
        let offsets = self.resource.zone_row_offsets_for_lookup();
        // Binary search for the zone containing `row`.
        let zi = offsets.partition_point(|&o| o <= row).saturating_sub(1);
        offsets
            .get(zi + 1)
            .copied()
            .unwrap_or_else(|| self.resource.total_rows())
    }
}

impl Operator for ZoneMapOperator {
    type GlobalState = ();
    type LocalState = ZoneMapOperatorState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new(
                "data",
                self.input_domain.clone(),
                self.output_columns,
            )],
            Some(OutputPortSpec::new(
                "out",
                self.output_domain.clone(),
                self.output_columns,
            )),
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
        Ok(ZoneMapOperatorState {
            sealed: false,
            last_seen_resource_version: 0,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let refined = &mut inputs[0];
        if self.resource.version() == 0 {
            // Zone map not yet published. Forward downstream demand
            // to the data subgraph but with a 0.5 selectivity prior:
            // rows are still `Required` for correctness — the data
            // subgraph must remain ready to produce — but in the
            // absence of pruning evidence the EV scheduler should
            // bias against actually doing the work yet, so the
            // zones subgraph (full selectivity) wins the priority
            // race and computes the prune mask first. Once the mask
            // is published, this function re-runs with a sharper
            // verdict per zone (selectivity 1.0 for kept zones, 0.0
            // for pruned).
            for iv in output.intervals() {
                let span = DomainSpan::new(iv.start, iv.end - iv.start);
                match iv.demand {
                    crate::RowDemand::NotNeeded => refined.not_needed_span(span),
                    crate::RowDemand::Needed | crate::RowDemand::Candidate => {
                        refined.require_span_with_selectivity(span, crate::Selectivity::HALF);
                    }
                    crate::RowDemand::Unknown => { /* leave Unknown */ }
                }
            }
            return Ok(());
        }
        // Resource published. Translate the downstream
        // requirement (in our output domain, layout-local) into
        // input requirements (also layout-local — the
        // ChunkConcat we feed has already converted to its own
        // local coords). For each downstream interval, walk the
        // zones overlapping it and split into Needed/NotNeeded
        // sub-intervals based on the resource's pruning mask.
        for iv in output.intervals() {
            // Only propagate intervals that downstream wants
            // (Needed/Candidate). NotNeeded passes through.
            // Unknown (default) is equivalent to "not in any
            // interval" — we don't synthesise demand for it.
            if iv.demand == crate::RowDemand::NotNeeded {
                refined.not_needed_span(DomainSpan::new(iv.start, iv.end - iv.start));
                continue;
            }
            // Translate iv to data-row coords (= layout-local +
            // bind-time row_range offset).
            let abs_start = self.data_row_offset + iv.start;
            let abs_end = self.data_row_offset + iv.end;
            // Walk zones overlapping [abs_start, abs_end). For
            // each zone, decide pruned vs kept.
            let mut row = abs_start;
            while row < abs_end {
                // Find the zone containing `row`.
                let one = row..row + 1;
                let pruned = self.resource.is_range_fully_pruned(one);
                // Find the zone's end so we can advance in one
                // step rather than per-row.
                let zone_end = self.zone_end_at(row).min(abs_end);
                let zone_local_start = row - self.data_row_offset;
                let zone_local_end = zone_end - self.data_row_offset;
                let span = DomainSpan::new(zone_local_start, zone_local_end - zone_local_start);
                if pruned {
                    refined.not_needed_span(span);
                } else {
                    refined.require_span(span);
                }
                row = zone_end;
            }
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // Poll the private `ZoneMapResource` and request a fresh
        // propagate pass only when the version has advanced. The
        // resource isn't a scheduler-managed `Resource`, so its
        // publishes do not surface as `DirtyCause::ResourceUpdated`
        // — instead we observe it here, on every wake of this
        // lane, and request propagation exactly when there's
        // something new to translate. After the zone-map is built
        // and stable the version stays put and this hook is a
        // no-op, so a query whose demand has stopped changing
        // stops paying for back-prop.
        let v = self.resource.version();
        if v > local.last_seen_resource_version {
            local.last_seen_resource_version = v;
            ctx.request_propagation();
        }
        let port = InputPortId::from_index(0);
        let peeked = ctx.peek(port);
        let finished = ctx.input_finished(port);
        if peeked.is_none() && !finished {
            return Ok(());
        }
        let useful_rows = peeked
            .as_ref()
            .map(|b| b.demand().true_count() as u64)
            .unwrap_or(0);
        let value = if useful_rows > 0 {
            WorkValue::required(useful_rows)
        } else {
            WorkValue::candidate(0, 0)
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            WorkClass::Emit,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::output_capacity(),
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
        if local.sealed {
            return Ok(WorkStatus::Finished);
        }
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }
        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            let span = batch.span();
            // Fast path: the batch is already a placeholder. After
            // back-prop has settled, the codes flat source emits
            // placeholder batches for pruned zones, so this is the
            // common case for pruned ranges. No AND needed.
            if batch.demand_all_false() {
                ctx.push(batch)?;
                return Ok(WorkStatus::Made);
            }
            let abs_start = self.data_row_offset + span.start();
            let abs_end = abs_start + span.len();
            let refined = self.resource.demand_for_range(abs_start..abs_end, 0);
            let out_batch = match refined {
                Some((_v, prune_demand)) => {
                    // Fast path: prune mask is uniformly all-true
                    // for this span (every overlapping zone is
                    // kept). The AND is a no-op — emit the batch
                    // unchanged. Avoids the bit-AND pass + the
                    // `with_demand` allocation.
                    if prune_demand.all_true() {
                        ctx.push(batch)?;
                        return Ok(WorkStatus::Made);
                    }
                    // Fast path: prune mask is uniformly all-false
                    // (every overlapping zone pruned). Emit a
                    // placeholder; no need to keep the array.
                    if prune_demand.all_false() {
                        ctx.push(Batch::placeholder(
                            span,
                            batch.dtype().clone(),
                        ))?;
                        return Ok(WorkStatus::Made);
                    }
                    // Mixed: AND the resource's demand into the
                    // batch's demand mask. Row domain is preserved.
                    use std::ops::BitAnd;
                    let combined = batch.demand().bitand(&prune_demand);
                    Batch::with_demand(span, batch.into_array(), combined)
                }
                None => batch,
            };
            ctx.push(out_batch)?;
            return Ok(WorkStatus::Made);
        }
        if ctx.input_finished(InputPortId::from_index(0)) {
            local.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}
