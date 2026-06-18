// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for zoned (zone-map) layouts: the canonical proof producer.
//!
//! Reading delegates straight to the data child. Pushed predicate nodes
//! expose zone-map evidence plans: per predicate, the falsification and
//! satisfaction rewrites ([`Expression::falsify`] /
//! [`Expression::satisfy`]) are evaluated over the zone map once per
//! query, and evidence walks the per-zone masks.
//!
//! Coverage is partial (plan 017 SP3): every zone proves its own span,
//! so a morsel misaligned with zone boundaries still gets evidence for
//! its interior zones — the v1 scan's whole-morsel verdict is just the
//! case where every overlapping zone agrees. Edge rows the statistics
//! cannot prove stay unknown and fall through to residual evaluation.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layout_v2::Layout;
use crate::layout_v2::Zoned;
use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::MIN_IS_TRUNCATED;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::scan::v2::evidence::EvidenceFragment;
use crate::scan::v2::evidence::PredicateEvidenceKind;
use crate::scan::v2::node::AggregateAnswer;
use crate::scan::v2::node::AggregatePlan;
use crate::scan::v2::node::AggregatePlanRef;
use crate::scan::v2::node::DynReadPlan;
use crate::scan::v2::node::EvidencePlan;
use crate::scan::v2::node::EvidencePlanRef;
use crate::scan::v2::node::EvidenceStateKey;
use crate::scan::v2::node::ExpandCtx;
use crate::scan::v2::node::FileReader;
use crate::scan::v2::node::PlanCtx;
use crate::scan::v2::node::PushCtx;
use crate::scan::v2::node::ReadPlan;
use crate::scan::v2::node::ReadPlanRef;
use crate::scan::v2::node::RowScope;
use crate::scan::v2::node::ScanNode;
use crate::scan::v2::node::ScanNodeRef;
use crate::scan::v2::node::ScanStateCache;
use crate::scan::v2::node::ScanStateRef;
use crate::scan::v2::node::StateCtx;
use crate::scan::v2::node::read_dense;
use crate::scan::v2::request::EvidenceRequest;
use crate::scan::v2::request::NodeRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_node(
    layout: Layout<Zoned>,
    req: &mut NodeRequest,
    cx: &ExpandCtx,
) -> VortexResult<ScanNodeRef> {
    let zones = layout.child(1)?;
    Ok(Arc::new(ZonedScanNode {
        // The data child preserves this node's rows: pass the
        // expansion request through.
        data: cx.expand(&layout.child(0)?, req)?,
        nzones: zones.row_count(),
        zones: cx.expand_free(&zones)?,
        column_dtype: layout.dtype().clone(),
        zone_len: layout.data().zone_len() as u64,
        row_count: layout.row_count(),
        present_stats: Arc::clone(layout.data().present_stats()),
    }))
}

/// Reads a zoned layout by delegating to its data child; produces
/// per-zone predicate evidence from the stats table.
pub struct ZonedScanNode {
    data: ScanNodeRef,
    /// The zones child (per-zone stats table), read through its own layout vtable.
    zones: ScanNodeRef,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    present_stats: Arc<[Stat]>,
}

/// Per-zone masks proven for one predicate.
struct PredicateMasks {
    /// `true` at zone `z`: the predicate is false for every row of `z`.
    all_false: Option<Mask>,
    /// `true` at zone `z`: the predicate is true for every row of `z`.
    all_true: Option<Mask>,
}

/// Per-query state: the child states plus the lazily decoded stats table,
/// the zone map, the per-predicate masks built from it, and the per-stat
/// columns prepared for range aggregation.
pub struct ZonedScanState {
    data: ScanStateRef,
    zones: ScanStateRef,
    /// The decoded per-zone stats table.
    table: Mutex<Option<Arc<StructArray>>>,
    zone_map: Mutex<Option<Arc<ZoneMap>>>,
    masks: Mutex<FxHashMap<Expression, Arc<PredicateMasks>>>,
    stat_columns: Mutex<FxHashMap<Stat, Option<Arc<StatColumn>>>>,
}

/// Planned evidence for one predicate over a zoned node.
struct ZonedEvidencePlan {
    zones_read: ReadPlanRef,
    zones_key: usize,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    present_stats: Arc<[Stat]>,
    predicate: Expression,
    falsifier: Option<Expression>,
    satisfier: Option<Expression>,
}

/// Planned ungrouped aggregate over a zoned node's root value.
struct ZonedAggregatePlan {
    node: Arc<ZonedScanNode>,
    zones_read: ReadPlanRef,
    funcs: Vec<AggregateFnRef>,
}

struct ZonedReadPlan {
    node: Arc<ZonedScanNode>,
    data: ReadPlanRef,
    zones: ReadPlanRef,
}

/// A pushed scalar expression through a zoned wrapper. Reads delegate to
/// the pushed data-child expression; evidence combines zone-map proof for
/// the expression with any child evidence for the same pushed value.
struct ZonedExprScanNode {
    data: ScanNodeRef,
    zones: ScanNodeRef,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    present_stats: Arc<[Stat]>,
    expr: Expression,
    falsifier: Option<Expression>,
    satisfier: Option<Expression>,
}

struct ZonedExprReadPlan {
    node: Arc<ZonedExprScanNode>,
    data: ReadPlanRef,
    zones: ReadPlanRef,
}

/// The zone coverage of one aggregate request: the requested rows, the
/// span of the zones lying wholly inside them, and those zones.
struct ZoneSpan {
    range: Range<u64>,
    interior: Range<u64>,
    zones: Range<usize>,
}

/// One stat's per-zone values prepared for vectorized range aggregation.
struct StatColumn {
    /// The stats-table column: one value per zone.
    values: ArrayRef,
    /// Per-zone validity; a null entry cannot answer exactly.
    valid: Mask,
    /// `true` at zones whose stored min/max is truncated — an outward
    /// bound, not the extremum.
    truncated: Option<Mask>,
}

impl StatColumn {
    /// Whether every zone in `zones` carries an exact value.
    fn all_exact(&self, zones: Range<usize>) -> bool {
        self.valid.slice(zones.clone()).all_true()
            && self
                .truncated
                .as_ref()
                .is_none_or(|truncated| truncated.slice(zones).all_false())
    }

    /// Whether zone `zone` carries an exact value.
    fn is_exact(&self, zone: usize) -> bool {
        self.valid.value(zone)
            && self
                .truncated
                .as_ref()
                .is_none_or(|truncated| !truncated.value(zone))
    }
}

impl ZonedScanState {
    /// The data child's state, for retention tests.
    #[allow(dead_code)]
    #[cfg(any(test, debug_assertions))]
    pub fn data_state(&self) -> &ScanStateRef {
        &self.data
    }
}

impl ZonedScanNode {
    /// The decoded per-zone stats table, read once per query. Concurrent
    /// decodes are benign (the segment fetch is shared; last-write-wins).
    async fn table(
        &self,
        zones_read: &dyn DynReadPlan,
        io: &FileReader,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<StructArray>> {
        if let Some(hit) = state.table.lock().clone() {
            return Ok(hit);
        }
        let zones = read_dense(zones_read, 0..self.nzones, io, state.zones.as_ref()).await?;
        let mut ctx = io.session().create_execution_ctx();
        let table = Arc::new(zones.execute::<StructArray>(&mut ctx)?);
        *state.table.lock() = Some(Arc::clone(&table));
        Ok(table)
    }

    /// One stat's per-zone column prepared for aggregation, built once
    /// per query directly over the stats table: the values, their
    /// validity, and (for min/max) the truncation flags.
    async fn stat_column(
        &self,
        stat: Stat,
        zones_read: &dyn DynReadPlan,
        io: &FileReader,
        state: &ZonedScanState,
    ) -> VortexResult<Option<Arc<StatColumn>>> {
        if let Some(hit) = state.stat_columns.lock().get(&stat) {
            return Ok(hit.clone());
        }
        let table = self.table(zones_read, io, state).await?;
        let mut ctx = io.session().create_execution_ctx();
        let column = match table.unmasked_field_by_name_opt(stat.name()) {
            None => None,
            Some(values) => {
                let values = values.clone();
                let valid = values.validity()?.execute_mask(values.len(), &mut ctx)?;
                let truncated = match stat {
                    Stat::Min => Some(MIN_IS_TRUNCATED),
                    Stat::Max => Some(MAX_IS_TRUNCATED),
                    _ => None,
                }
                .map(|flag| match table.unmasked_field_by_name_opt(flag) {
                    Some(flags) => flags.clone().execute::<Mask>(&mut ctx),
                    // No recorded flags: nothing can be proven exact.
                    None => Ok(Mask::new_true(values.len())),
                })
                .transpose()?;
                Some(Arc::new(StatColumn {
                    values,
                    valid,
                    truncated,
                }))
            }
        };
        state.stat_columns.lock().insert(stat, column.clone());
        Ok(column)
    }

    /// Answer one aggregate function over `range` from the per-zone
    /// statistics: a partial for the answerable interior zones, residual
    /// spans for edge fragments and unanswerable zones. `None`: nothing
    /// covered, the caller owns the whole range.
    async fn aggregate_one(
        &self,
        span: &ZoneSpan,
        func: &AggregateFnRef,
        zones_read: &dyn DynReadPlan,
        io: &FileReader,
        state: &ZonedScanState,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<AggregateAnswer>> {
        let ZoneSpan {
            range,
            interior,
            zones,
        } = span;
        let zones = zones.clone();
        // The stat that answers `func` per zone: count(column) derives
        // from null_count; everything else maps through its stored stat.
        let is_count = func.is::<Count>();
        let stat = if is_count {
            Stat::NullCount
        } else {
            let Some(stat) = Stat::from_aggregate_fn(func) else {
                return Ok(None);
            };
            stat
        };
        let Some(partial_dtype) = func.state_dtype(&self.column_dtype) else {
            return Ok(None);
        };
        let Some(col) = self.stat_column(stat, zones_read, io, state).await? else {
            return Ok(None);
        };

        // Edge fragments and unanswerable zones are the caller's to read,
        // adjacent spans merged.
        let mut residual: Vec<Range<u64>> = Vec::new();
        let mut push_residual = |span: Range<u64>| match residual.last_mut() {
            Some(last) if last.end == span.start => last.end = span.end,
            _ => residual.push(span),
        };
        if range.start < interior.start {
            push_residual(range.start..interior.start);
        }

        let mut accumulator = func.accumulator(&self.column_dtype)?;
        let mut contributed = false;
        let mut covered = false;
        if col.all_exact(zones.clone()) {
            // Every zone answers: reduce the stats column slice with the
            // fn's own kernels. Combining per-zone partials of a stat fn
            // is the fn itself over the per-zone values — sum of sums,
            // min of mins — and count(column) is the covered row count
            // minus the summed per-zone null counts.
            let values = col.values.slice(zones)?;
            let partial = if is_count {
                let mut nulls = Sum.bind(EmptyOptions).accumulator(values.dtype())?;
                nulls.accumulate(&values, ctx)?;
                let nulls = nulls
                    .flush()?
                    .as_primitive()
                    .typed_value::<u64>()
                    .ok_or_else(|| vortex_err!("summed null counts must be non-null"))?;
                let rows = interior.end - interior.start;
                Scalar::primitive(rows - nulls, Nullability::NonNullable)
            } else {
                let mut combine = func.accumulator(values.dtype())?;
                combine.accumulate(&values, ctx)?;
                combine.flush()?
            };
            accumulator.combine_partials(partial.cast(&partial_dtype)?)?;
            contributed = true;
            covered = true;
        } else {
            // Rare: some zone's stat is null or truncated. Walk the
            // zones, answering what the stats prove and leaving the rest
            // to the caller.
            let null_counts = match stat {
                Stat::NullCount => Some(Arc::clone(&col)),
                _ => {
                    self.stat_column(Stat::NullCount, zones_read, io, state)
                        .await?
                }
            };
            let zone_nulls = |zone: usize, ctx: &mut ExecutionCtx| -> VortexResult<Option<u64>> {
                match &null_counts {
                    Some(nulls) if nulls.valid.value(zone) => Ok(nulls
                        .values
                        .execute_scalar(zone, ctx)?
                        .as_primitive()
                        .typed_value::<u64>()),
                    _ => Ok(None),
                }
            };
            for zone in zones {
                let span = self.zone_span(zone);
                if col.is_exact(zone) {
                    let partial = if is_count {
                        let nulls = zone_nulls(zone, ctx)?
                            .ok_or_else(|| vortex_err!("null count must be non-null"))?;
                        let rows = span.end - span.start;
                        Scalar::primitive(rows - nulls, Nullability::NonNullable)
                    } else {
                        col.values.execute_scalar(zone, ctx)?
                    };
                    accumulator.combine_partials(partial.cast(&partial_dtype)?)?;
                    contributed = true;
                    covered = true;
                    continue;
                }
                // A null or truncated stat over a provably all-null zone
                // is still covered: the aggregate of zero non-null values
                // contributes nothing. (Not a null partial — for sum, a
                // null partial means overflow.)
                if zone_nulls(zone, ctx)? == Some(span.end - span.start) {
                    covered = true;
                    continue;
                }
                push_residual(span);
            }
        }
        if !covered {
            return Ok(None);
        }
        if interior.end < range.end {
            push_residual(interior.end..range.end);
        }
        Ok(Some(AggregateAnswer {
            partial: contributed.then(|| accumulator.flush()).transpose()?,
            residual,
        }))
    }

    /// The row span of zone `zone` (the final zone may be short).
    fn zone_span(&self, zone: usize) -> Range<u64> {
        let start = zone as u64 * self.zone_len;
        start..(start + self.zone_len).min(self.row_count)
    }

    /// The zones lying entirely inside `range`.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "zone counts stay far below usize::MAX on supported targets"
    )]
    fn interior_zones(&self, range: &Range<u64>) -> Range<usize> {
        let start = (range.start.div_ceil(self.zone_len)) as usize;
        // The final zone may be short: it is inside the range iff the
        // range reaches the end of the layout.
        let end = if range.end == self.row_count {
            self.nzones as usize
        } else {
            (range.end / self.zone_len) as usize
        };
        start..end.max(start)
    }

    /// Answer aggregates from per-zone statistics over every
    /// row of `range`. Filtered aggregate pushdown is intentionally out
    /// of scope for this simple stats path.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        funcs: &'a [AggregateFnRef],
        zones_read: &'a dyn DynReadPlan,
        io: &'a FileReader,
        state: &'a ZonedScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>> {
        Box::pin(async move {
            if self.zone_len == 0 {
                // Legacy files without a recorded zone length.
                return Ok(None);
            }
            let zones = self.interior_zones(&range);
            if zones.is_empty() {
                return Ok(None);
            }
            let span = ZoneSpan {
                range: range.clone(),
                interior: self.zone_span(zones.start).start..self.zone_span(zones.end - 1).end,
                zones,
            };
            let mut ctx = io.session().create_execution_ctx();
            let mut answers = Vec::with_capacity(funcs.len());
            let mut covered_any = false;
            for func in funcs {
                match self
                    .aggregate_one(&span, func, zones_read, io, state, &mut ctx)
                    .await?
                {
                    Some(answer) => {
                        covered_any = true;
                        answers.push(answer);
                    }
                    None => answers.push(AggregateAnswer {
                        partial: None,
                        residual: vec![range.clone()],
                    }),
                }
            }
            // When no function is answerable, let the caller issue plain
            // whole-range reads instead.
            if !covered_any {
                return Ok(None);
            }
            Ok(Some(answers))
        })
    }
}

impl ZonedEvidencePlan {
    async fn table(
        &self,
        io: &FileReader,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<StructArray>> {
        if let Some(hit) = state.table.lock().clone() {
            return Ok(hit);
        }
        let zones = read_dense(
            self.zones_read.as_ref(),
            0..self.nzones,
            io,
            state.zones.as_ref(),
        )
        .await?;
        let mut ctx = io.session().create_execution_ctx();
        let table = Arc::new(zones.execute::<StructArray>(&mut ctx)?);
        *state.table.lock() = Some(Arc::clone(&table));
        Ok(table)
    }

    async fn zone_map(
        &self,
        io: &FileReader,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<ZoneMap>> {
        if let Some(hit) = state.zone_map.lock().clone() {
            return Ok(hit);
        }
        let table = self.table(io, state).await?;
        let zone_map = Arc::new(ZoneMap::try_new(
            self.column_dtype.clone(),
            (*table).clone(),
            Arc::clone(&self.present_stats),
            self.zone_len,
            self.row_count,
        )?);
        *state.zone_map.lock() = Some(Arc::clone(&zone_map));
        Ok(zone_map)
    }

    async fn predicate_masks(
        &self,
        io: &FileReader,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<PredicateMasks>> {
        if let Some(hit) = state.masks.lock().get(&self.predicate) {
            return Ok(Arc::clone(hit));
        }
        let zone_map = self.zone_map(io, state).await?;
        let session = io.session();
        let all_false = self
            .falsifier
            .as_ref()
            .map(|falsification| zone_map.prune(falsification, session))
            .transpose()?;
        let all_true = self
            .satisfier
            .as_ref()
            .map(|satisfaction| zone_map.prune(satisfaction, session))
            .transpose()?;
        let masks = Arc::new(PredicateMasks {
            all_false,
            all_true,
        });
        state
            .masks
            .lock()
            .insert(self.predicate.clone(), Arc::clone(&masks));
        Ok(masks)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "zone counts stay far below usize::MAX on supported targets"
    )]
    fn zone_range(&self, range: &Range<u64>) -> Range<usize> {
        let start = (range.start / self.zone_len) as usize;
        let end = range.end.div_ceil(self.zone_len) as usize;
        start..end.min(self.nzones as usize)
    }

    fn zone_span(&self, zone: usize) -> Range<u64> {
        let start = zone as u64 * self.zone_len;
        start..(start + self.zone_len).min(self.row_count)
    }
}

impl EvidencePlan for ZonedEvidencePlan {
    type State = ZonedScanState;

    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ZonedScanState> {
        let mut cache = ScanStateCache::default();
        let mut cx = StateCtx::new(ctx, &mut cache);
        Ok(ZonedScanState {
            data: Arc::new(()),
            zones: self.zones_read.init_state(&mut cx)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
        state: &'a ZonedScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        Box::pin(async move {
            let mut fragments = Vec::new();
            if self.zone_len > 0 && (self.falsifier.is_some() || self.satisfier.is_some()) {
                let masks = self.predicate_masks(io, state).await?;
                let zones = self.zone_range(&req.range);
                let mut run: Option<(Range<u64>, bool)> = None;
                for zone in zones {
                    let all_false = masks
                        .all_false
                        .as_ref()
                        .is_some_and(|mask| mask.value(zone));
                    let all_true =
                        !all_false && masks.all_true.as_ref().is_some_and(|mask| mask.value(zone));
                    let span = self.zone_span(zone);
                    match (&mut run, all_false || all_true) {
                        (Some((rows, false_run)), true) if *false_run == all_false => {
                            rows.end = span.end;
                        }
                        (current, proven) => {
                            if let Some((rows, false_run)) = current.take() {
                                fragments.push(fragment(rows, false_run));
                            }
                            if proven {
                                *current = Some((span, all_false));
                            }
                        }
                    }
                }
                if let Some((rows, false_run)) = run {
                    fragments.push(fragment(rows, false_run));
                }
            }
            Ok(fragments)
        })
    }

    fn segment_requests(
        &self,
        _req: &EvidenceRequest<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if self.zone_len == 0 || (self.falsifier.is_none() && self.satisfier.is_none()) {
            return Ok(SegmentRequests::none());
        }
        let selection = Mask::new_true(
            usize::try_from(self.nzones)
                .map_err(|_| vortex_err!("zoned stats length exceeds usize"))?,
        );
        self.zones_read.segment_requests(
            0..self.nzones,
            RowScope::selected(&selection),
            state.zones.as_ref(),
            cx,
        )
    }

    fn state_cache_key(&self) -> Option<EvidenceStateKey> {
        Some(EvidenceStateKey::new::<Self>(self.zones_key))
    }

    fn recheck_before_projection(&self) -> bool {
        true
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned")
    }
}

impl ScanNode for ZonedScanNode {
    type State = ZonedScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ZonedScanState> {
        Ok(ZonedScanState {
            data: cx.init_node(&self.data)?,
            zones: cx.init_node(&self.zones)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let data = Arc::clone(&self.data)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("zoned data child did not produce a read plan"))?;
        let zones = Arc::clone(&self.zones)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("zoned stats child did not produce a read plan"))?;
        Ok(Some(Arc::new(ZonedReadPlan {
            node: self,
            data,
            zones,
        })))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
        if is_root(expr) {
            return Ok(Some(self));
        }
        let Some(data) = Arc::clone(&self.data).try_push_expr(expr, cx)? else {
            return Ok(None);
        };
        let is_predicate = matches!(expr.return_dtype(&self.column_dtype)?, DType::Bool(_));
        let (falsifier, satisfier) = if self.zone_len > 0 && is_predicate {
            (
                expr.falsify(&self.column_dtype, cx.session())?,
                expr.satisfy(&self.column_dtype, cx.session())?,
            )
        } else {
            (None, None)
        };
        Ok(Some(Arc::new(ZonedExprScanNode {
            data,
            zones: Arc::clone(&self.zones),
            nzones: self.nzones,
            column_dtype: self.column_dtype.clone(),
            zone_len: self.zone_len,
            row_count: self.row_count,
            present_stats: Arc::clone(&self.present_stats),
            expr: expr.clone(),
            falsifier,
            satisfier,
        })))
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        let mut plans = Arc::clone(&self.data).plan_evidence(cx)?;
        let predicate = root();
        let is_predicate = matches!(predicate.return_dtype(&self.column_dtype)?, DType::Bool(_));
        let (falsifier, satisfier) = if self.zone_len > 0 && is_predicate {
            (
                predicate.falsify(&self.column_dtype, cx.session())?,
                predicate.satisfy(&self.column_dtype, cx.session())?,
            )
        } else {
            (None, None)
        };
        if falsifier.is_some() || satisfier.is_some() {
            let zones_key = Arc::as_ptr(&self.zones) as *const () as usize;
            let zones_read = Arc::clone(&self.zones)
                .plan_read(cx)?
                .ok_or_else(|| vortex_err!("zoned stats child did not produce a read plan"))?;
            plans.insert(
                0,
                Arc::new(ZonedEvidencePlan {
                    zones_read,
                    zones_key,
                    nzones: self.nzones,
                    column_dtype: self.column_dtype.clone(),
                    zone_len: self.zone_len,
                    row_count: self.row_count,
                    present_stats: Arc::clone(&self.present_stats),
                    predicate,
                    falsifier,
                    satisfier,
                }),
            );
        }
        Ok(plans)
    }

    fn plan_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>> {
        if funcs.is_empty() {
            return Ok(None);
        }
        let zones_read = Arc::clone(&self.zones)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("zoned stats child did not produce a read plan"))?;
        Ok(Some(Arc::new(ZonedAggregatePlan {
            node: self,
            zones_read,
            funcs: funcs.to_vec(),
        })))
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.data.split_hints()
    }

    /// The data child holds the real memory and releases with this
    /// node's row domain. The decoded stats table, zone map, and
    /// per-predicate masks stay: they are small, deliberately per-query,
    /// and consulted for every remaining morsel.
    fn release(&self, frontier: u64, state: &ZonedScanState) -> VortexResult<()> {
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned:")?;
        self.data.fmt_chain(f)
    }
}

impl ReadPlan for ZonedReadPlan {
    type State = ZonedScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(ZonedScanState {
            data: self.data.init_state(cx)?,
            zones: self.zones.init_state(cx)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        self.data
            .read_scoped(range, rows, io, state.data.as_ref(), local)
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.data
            .segment_requests(range, rows, state.data.as_ref(), cx)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl AggregatePlan for ZonedAggregatePlan {
    type State = ZonedScanState;

    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ZonedScanState> {
        let mut cache = ScanStateCache::default();
        let mut cx = StateCtx::new(ctx, &mut cache);
        Ok(ZonedScanState {
            data: Arc::new(()),
            zones: self.zones_read.init_state(&mut cx)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ZonedScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>> {
        self.node
            .aggregate_partial(range, &self.funcs, self.zones_read.as_ref(), io, state)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned")
    }
}

impl ScanNode for ZonedExprScanNode {
    type State = ZonedScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(ZonedScanState {
            data: cx.init_node(&self.data)?,
            zones: cx.init_node(&self.zones)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let data = Arc::clone(&self.data).plan_read(cx)?.ok_or_else(|| {
            vortex_err!("zoned expression data child did not produce a read plan")
        })?;
        let zones = Arc::clone(&self.zones)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("zoned stats child did not produce a read plan"))?;
        Ok(Some(Arc::new(ZonedExprReadPlan {
            node: self,
            data,
            zones,
        })))
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        let mut plans = Arc::clone(&self.data).plan_evidence(cx)?;
        if self.falsifier.is_some() || self.satisfier.is_some() {
            let zones_key = Arc::as_ptr(&self.zones) as *const () as usize;
            let zones_read = Arc::clone(&self.zones)
                .plan_read(cx)?
                .ok_or_else(|| vortex_err!("zoned stats child did not produce a read plan"))?;
            plans.insert(
                0,
                Arc::new(ZonedEvidencePlan {
                    zones_read,
                    zones_key,
                    nzones: self.nzones,
                    column_dtype: self.column_dtype.clone(),
                    zone_len: self.zone_len,
                    row_count: self.row_count,
                    present_stats: Arc::clone(&self.present_stats),
                    predicate: self.expr.clone(),
                    falsifier: self.falsifier.clone(),
                    satisfier: self.satisfier.clone(),
                }),
            );
        }
        Ok(plans)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned_expr({})", self.expr)
    }
}

impl ReadPlan for ZonedExprReadPlan {
    type State = ZonedScanState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(ZonedScanState {
            data: self.data.init_state(cx)?,
            zones: self.zones.init_state(cx)?,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        })
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        self.data
            .read_scoped(range, rows, io, state.data.as_ref(), local)
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.data
            .segment_requests(range, rows, state.data.as_ref(), cx)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

fn fragment(rows: Range<u64>, all_false: bool) -> EvidenceFragment {
    EvidenceFragment::new(
        rows,
        if all_false {
            PredicateEvidenceKind::AllFalse
        } else {
            PredicateEvidenceKind::AllTrue
        },
    )
}
