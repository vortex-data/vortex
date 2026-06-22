// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for zoned (zone-map) layouts: the canonical proof producer.
//!
//! Reading delegates straight to the data child. Pushed predicate nodes
//! expose zone-map prepared evidence: per predicate, the falsification and
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
use vortex_array::aggregate_fn::NumericalAggregateOpts;
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
use vortex_scan::plan::AggregateAnswer;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedAggregate;
use vortex_scan::plan::PreparedAggregateRef;
use vortex_scan::plan::PreparedEvidence;
use vortex_scan::plan::PreparedEvidenceRef;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStateKey;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadContext;
use vortex_scan::plan::RowScope;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::plan::downcast_state;
use vortex_scan::plan::evidence::EvidenceFragment;
use vortex_scan::plan::evidence::PredicateEvidenceKind;
use vortex_scan::plan::read_dense;
use vortex_scan::plan::request::EvidenceRequest;
use vortex_scan::plan::request::ScanRequest;
use vortex_session::VortexSession;

use crate::layout_v2::Layout;
use crate::layout_v2::VTable;
use crate::layout_v2::ZonedData;
use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::MIN_IS_TRUNCATED;
use crate::layouts::zoned::ZoneMapSchema;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_plan<V: VTable<LayoutData = ZonedData>>(
    layout: Layout<V>,
    req: &mut ScanRequest,
    session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    let zones = layout.child(1)?;
    Ok(Arc::new(ZonedScanPlan {
        // The data child preserves this node's rows: pass the
        // expansion request through.
        data: layout.child(0)?.new_scan_plan(req, session)?,
        nzones: zones.row_count(),
        zones: zones.new_scan_plan(&mut ScanRequest::empty(), session)?,
        column_dtype: layout.dtype().clone(),
        zone_len: layout.data().zone_len() as u64,
        row_count: layout.row_count(),
        zone_map_schema: layout.data().zone_map_schema().clone(),
        aggregate_fns: Arc::clone(layout.data().aggregate_fns()),
    }))
}

/// Reads a zoned layout by delegating to its data child; produces
/// per-zone predicate evidence from the stats table.
pub struct ZonedScanPlan {
    data: ScanPlanRef,
    /// The zones child (per-zone stats table), read through its own layout vtable.
    zones: ScanPlanRef,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    zone_map_schema: ZoneMapSchema,
    aggregate_fns: Arc<[AggregateFnRef]>,
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
    /// The decoded per-zone stats table.
    table: Mutex<Option<Arc<StructArray>>>,
    zone_map: Mutex<Option<Arc<ZoneMap>>>,
    masks: Mutex<FxHashMap<Expression, Arc<PredicateMasks>>>,
    stat_columns: Mutex<FxHashMap<Stat, Option<Arc<StatColumn>>>>,
}

/// Planned evidence for one predicate over a zoned node.
struct ZonedPreparedEvidence {
    state: Arc<ZonedScanState>,
    zones_read: PreparedReadRef,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    zone_map_schema: ZoneMapSchema,
    aggregate_fns: Arc<[AggregateFnRef]>,
    predicate: Expression,
    falsifier: Option<Expression>,
    satisfier: Option<Expression>,
}

/// Planned ungrouped aggregate over a zoned node's root value.
struct ZonedPreparedAggregate {
    node: Arc<ZonedScanPlan>,
    state: Arc<ZonedScanState>,
    zones_read: PreparedReadRef,
    funcs: Vec<AggregateFnRef>,
}

struct ZonedPreparedRead {
    node: Arc<ZonedScanPlan>,
    data: PreparedReadRef,
}

/// A pushed scalar expression through a zoned wrapper. Reads delegate to
/// the pushed data-child expression; evidence combines zone-map proof for
/// the expression with any child evidence for the same pushed value.
struct ZonedExprScanPlan {
    data: ScanPlanRef,
    zones: ScanPlanRef,
    nzones: u64,
    column_dtype: DType,
    zone_len: u64,
    row_count: u64,
    zone_map_schema: ZoneMapSchema,
    aggregate_fns: Arc<[AggregateFnRef]>,
    expr: Expression,
    falsifier: Option<Expression>,
    satisfier: Option<Expression>,
}

struct ZonedExprPreparedRead {
    node: Arc<ZonedExprScanPlan>,
    data: PreparedReadRef,
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

impl ZonedScanPlan {
    fn shared_zone_state(&self, cx: &mut PrepareCtx) -> VortexResult<Arc<ZonedScanState>> {
        let key =
            PreparedStateKey::new::<ZonedScanState>(Arc::as_ptr(&self.zones) as *const () as usize);
        cx.shared_state(key, || Ok(Self::empty_state()))
    }

    fn empty_state_with_data(data: ScanStateRef) -> ZonedScanState {
        ZonedScanState {
            data,
            table: Mutex::new(None),
            zone_map: Mutex::new(None),
            masks: Mutex::new(FxHashMap::default()),
            stat_columns: Mutex::new(FxHashMap::default()),
        }
    }

    fn empty_state() -> ZonedScanState {
        Self::empty_state_with_data(Arc::new(()))
    }

    /// The decoded per-zone stats table, read once per query. Concurrent
    /// decodes are benign (the segment fetch is shared; last-write-wins).
    async fn table(
        &self,
        zones_read: &PreparedReadRef,
        io: &ReadContext,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<StructArray>> {
        if let Some(hit) = state.table.lock().clone() {
            return Ok(hit);
        }
        let zones = read_dense(zones_read, 0..self.nzones, io).await?;
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
        zones_read: &PreparedReadRef,
        io: &ReadContext,
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
        zones_read: &PreparedReadRef,
        io: &ReadContext,
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
                let mut nulls = Sum
                    .bind(NumericalAggregateOpts::default())
                    .accumulator(values.dtype())?;
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
        zones_read: &'a PreparedReadRef,
        io: &'a ReadContext,
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

impl ZonedPreparedEvidence {
    async fn table(
        &self,
        io: &ReadContext,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<StructArray>> {
        if let Some(hit) = state.table.lock().clone() {
            return Ok(hit);
        }
        let zones = read_dense(&self.zones_read, 0..self.nzones, io).await?;
        let mut ctx = io.session().create_execution_ctx();
        let table = Arc::new(zones.execute::<StructArray>(&mut ctx)?);
        *state.table.lock() = Some(Arc::clone(&table));
        Ok(table)
    }

    async fn zone_map(
        &self,
        io: &ReadContext,
        state: &ZonedScanState,
    ) -> VortexResult<Arc<ZoneMap>> {
        if let Some(hit) = state.zone_map.lock().clone() {
            return Ok(hit);
        }
        let table = self.table(io, state).await?;
        let zone_map = match &self.zone_map_schema {
            ZoneMapSchema::AggregateFns(_) => ZoneMap::try_new(
                self.column_dtype.clone(),
                (*table).clone(),
                Arc::clone(&self.aggregate_fns),
                self.zone_len,
                self.row_count,
            )?,
            ZoneMapSchema::LegacyStats(_) => {
                // Legacy stats-table dtypes are validated by child dtype construction.
                // SAFETY: V2 layout child deserialization checked the legacy stats-table dtype.
                unsafe {
                    ZoneMap::new_unchecked(
                        self.column_dtype.clone(),
                        (*table).clone(),
                        Arc::clone(&self.aggregate_fns),
                        self.zone_len,
                        self.row_count,
                    )
                }
            }
        };
        let zone_map = Arc::new(zone_map);
        *state.zone_map.lock() = Some(Arc::clone(&zone_map));
        Ok(zone_map)
    }

    async fn predicate_masks(
        &self,
        io: &ReadContext,
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

impl PreparedEvidence for ZonedPreparedEvidence {
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a ReadContext,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        Box::pin(async move {
            let mut fragments = Vec::new();
            if self.zone_len > 0 && (self.falsifier.is_some() || self.satisfier.is_some()) {
                let masks = self.predicate_masks(io, &self.state).await?;
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
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if self.zone_len == 0 || (self.falsifier.is_none() && self.satisfier.is_none()) {
            return Ok(SegmentRequests::none());
        }
        let selection = Mask::new_true(
            usize::try_from(self.nzones)
                .map_err(|_| vortex_err!("zoned stats length exceeds usize"))?,
        );
        self.zones_read
            .segment_requests(0..self.nzones, RowScope::selected(&selection), cx)
    }

    fn recheck_before_projection(&self) -> bool {
        true
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned")
    }
}

impl ScanPlan for ZonedScanPlan {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(Self::empty_state_with_data(
            cx.init_plan(&self.data)?,
        )))
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let data = Arc::clone(&self.data)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("zoned data child did not produce a prepared read"))?;
        Ok(Some(Arc::new(ZonedPreparedRead { node: self, data })))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
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
        Ok(Some(Arc::new(ZonedExprScanPlan {
            data,
            zones: Arc::clone(&self.zones),
            nzones: self.nzones,
            column_dtype: self.column_dtype.clone(),
            zone_len: self.zone_len,
            row_count: self.row_count,
            zone_map_schema: self.zone_map_schema.clone(),
            aggregate_fns: Arc::clone(&self.aggregate_fns),
            expr: expr.clone(),
            falsifier,
            satisfier,
        })))
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        let mut plans = Arc::clone(&self.data).prepare_evidence(cx)?;
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
            let state = self.shared_zone_state(cx)?;
            let zones_read = Arc::clone(&self.zones)
                .prepare_read(cx)?
                .ok_or_else(|| vortex_err!("zoned stats child did not produce a prepared read"))?;
            plans.insert(
                0,
                Arc::new(ZonedPreparedEvidence {
                    state,
                    zones_read,
                    nzones: self.nzones,
                    column_dtype: self.column_dtype.clone(),
                    zone_len: self.zone_len,
                    row_count: self.row_count,
                    zone_map_schema: self.zone_map_schema.clone(),
                    aggregate_fns: Arc::clone(&self.aggregate_fns),
                    predicate,
                    falsifier,
                    satisfier,
                }),
            );
        }
        Ok(plans)
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        if funcs.is_empty() {
            return Ok(None);
        }
        let zones_read = Arc::clone(&self.zones)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("zoned stats child did not produce a prepared read"))?;
        let state = self.shared_zone_state(cx)?;
        Ok(Some(Arc::new(ZonedPreparedAggregate {
            node: self,
            state,
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
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ZonedScanState>(state)?;
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned:")?;
        self.data.fmt_chain(f)
    }
}

impl PreparedRead for ZonedPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a ReadContext,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        self.data.read_scoped(range, rows, io, local)
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.data.segment_requests(range, rows, cx)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.data.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl PreparedAggregate for ZonedPreparedAggregate {
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a ReadContext,
        _state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>> {
        self.node
            .aggregate_partial(range, &self.funcs, &self.zones_read, io, &self.state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned")
    }
}

impl ScanPlan for ZonedExprScanPlan {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ZonedScanPlan::empty_state_with_data(
            cx.init_plan(&self.data)?,
        )))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let data = Arc::clone(&self.data).prepare_read(cx)?.ok_or_else(|| {
            vortex_err!("zoned expression data child did not produce a prepared read")
        })?;
        Ok(Some(Arc::new(ZonedExprPreparedRead { node: self, data })))
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        let mut plans = Arc::clone(&self.data).prepare_evidence(cx)?;
        if self.falsifier.is_some() || self.satisfier.is_some() {
            let key = PreparedStateKey::new::<ZonedScanState>(
                Arc::as_ptr(&self.zones) as *const () as usize,
            );
            let state = cx.shared_state(key, || Ok(ZonedScanPlan::empty_state()))?;
            let zones_read = Arc::clone(&self.zones)
                .prepare_read(cx)?
                .ok_or_else(|| vortex_err!("zoned stats child did not produce a prepared read"))?;
            plans.insert(
                0,
                Arc::new(ZonedPreparedEvidence {
                    state,
                    zones_read,
                    nzones: self.nzones,
                    column_dtype: self.column_dtype.clone(),
                    zone_len: self.zone_len,
                    row_count: self.row_count,
                    zone_map_schema: self.zone_map_schema.clone(),
                    aggregate_fns: Arc::clone(&self.aggregate_fns),
                    predicate: self.expr.clone(),
                    falsifier: self.falsifier.clone(),
                    satisfier: self.satisfier.clone(),
                }),
            );
        }
        Ok(plans)
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ZonedScanState>(state)?;
        self.data.release(frontier, state.data.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zoned_expr({})", self.expr)
    }
}

impl PreparedRead for ZonedExprPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a ReadContext,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        self.data.read_scoped(range, rows, io, local)
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.data.segment_requests(range, rows, cx)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.data.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
