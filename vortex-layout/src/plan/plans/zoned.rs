// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::EmptyMetadata;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Selection;
use crate::plan::exec::ZonePruneNode;
use crate::plan::optimizer::PlanParentReduceRule;

const DATA: usize = 0;
const ZONES: usize = 1;

/// Data summarised by per-zone statistics.
///
/// A zoned plan produces its data child's rows: it adds nothing to an execution. Its value is
/// the zone table beside the data, which a [`Query`](crate::plan::Query) uses to skip the zones a
/// conjunct cannot match. The optimizer pushes an expression over the plan into its data; a
/// boolean one is kept as the plan's predicate, and [`ZonedPlan::pruning_plan`] turns it into a
/// plan that tells, for every row, whether the row's zone may hold a match, from the zone table
/// alone.
///
/// The zone table, once read, and the proof of each predicate over it are kept on the plan and
/// shared by every plan derived from it, so a file's zones are read and proven once however many
/// queries and splits run over them.
#[derive(Clone, Debug)]
pub struct Zoned;

/// A plan over zoned data, or a pruning plan derived from one.
pub type ZonedPlan = Plan<Zoned>;

#[derive(Clone)]
enum Kind {
    /// The data, with the predicate over the column it evaluates when it is a conjunct.
    Data { predicate: Option<BoundExpression> },
    /// Whether each row's zone may hold a row passing the predicate.
    Prune { predicate: BoundExpression },
}

/// Zoned-plan-specific data.
#[derive(Clone)]
pub struct ZonedData {
    zone_len: u64,
    /// The dtype of the column the zones summarise.
    column_dtype: DType,
    aggregate_fns: Arc<[AggregateFnRef]>,
    kind: Kind,
    cache: Arc<ZoneCache>,
}

impl fmt::Debug for ZonedData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("ZonedData");
        s.field("zone_len", &self.zone_len)
            .field("column_dtype", &self.column_dtype)
            .field("aggregates", &self.aggregate_fns.len());
        match &self.kind {
            Kind::Data { predicate } => s.field("predicate", predicate),
            Kind::Prune { predicate } => s.field("prune", predicate),
        };
        s.finish()
    }
}

/// The zone table of a zoned plan once it has been read, and the proof of each predicate over
/// it, shared by every plan derived from the zoned plan.
pub(crate) struct ZoneCache {
    zone_map: OnceLock<ZoneMap>,
    /// The proof of each predicate, or `None` when no statistic proves it false.
    proofs: Mutex<FxHashMap<ExactBoundExpr, Option<Arc<Proof>>>>,
}

impl ZoneCache {
    fn new() -> Self {
        Self {
            zone_map: OnceLock::new(),
            proofs: Mutex::new(FxHashMap::default()),
        }
    }

    /// The zone table, if an execution has read it.
    pub(crate) fn zone_map(&self) -> Option<&ZoneMap> {
        self.zone_map.get()
    }

    /// Keeps the zone table an execution read, unless another did first.
    pub(crate) fn set_zone_map(&self, zone_map: ZoneMap) -> &ZoneMap {
        self.zone_map.get_or_init(|| zone_map)
    }

    /// The proof of `predicate` from statistics, built once per predicate, or `None` when no
    /// statistic proves it false anywhere.
    pub(crate) fn proof(
        &self,
        predicate: &BoundExpression,
        session: &VortexSession,
    ) -> Option<Arc<Proof>> {
        let key = ExactBoundExpr(predicate.clone());
        if let Some(proof) = self.proofs.lock().get(&key) {
            return proof.clone();
        }
        // A predicate no statistic can falsify is not a failure: it is just not prunable.
        let proof = predicate.falsify(session).ok().flatten().map(|falsifier| {
            Arc::new(Proof {
                dynamic: DynamicExprUpdates::new(predicate),
                falsifier,
                pruned: Mutex::new(None),
            })
        });
        self.proofs.lock().entry(key).or_insert(proof).clone()
    }
}

/// An expression that proves, from a zone's statistics, that the zone holds no row passing a
/// predicate, and the zones it proved that for.
pub(crate) struct Proof {
    falsifier: BoundExpression,
    /// Set when the predicate compares against a value that may change between executions, in
    /// which case the zones are proven again on every execution.
    dynamic: Option<DynamicExprUpdates>,
    pruned: Mutex<Option<Mask>>,
}

impl Proof {
    /// The zones that hold no passing row: `true` for a zone that can be skipped.
    pub(crate) fn pruned(&self, zone_map: &ZoneMap, session: &VortexSession) -> VortexResult<Mask> {
        if self.dynamic.is_some() {
            return zone_map.prune(&self.falsifier, session);
        }
        if let Some(pruned) = &*self.pruned.lock() {
            return Ok(pruned.clone());
        }
        let pruned = zone_map.prune(&self.falsifier, session)?;
        *self.pruned.lock() = Some(pruned.clone());
        Ok(pruned)
    }
}

impl ZonedPlan {
    /// A plan over `data`, summarised by the zone table `zones`, whose rows hold the results of
    /// `aggregate_fns` over consecutive zones of `zone_len` rows.
    pub(crate) fn from_children(
        dtype: DType,
        row_count: u64,
        children: PlanChildren,
        zone_len: u64,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> Self {
        PlanParts {
            vtable: Zoned,
            dtype: dtype.clone(),
            row_count,
            children,
            data: ZonedData {
                zone_len,
                column_dtype: dtype,
                aggregate_fns,
                kind: Kind::Data { predicate: None },
                cache: Arc::new(ZoneCache::new()),
            },
        }
        .into_typed()
    }

    /// Rows per zone.
    pub fn zone_len(&self) -> u64 {
        self.data().zone_len
    }

    /// The aggregates whose results the zone table holds, in field order.
    pub fn aggregate_fns(&self) -> &Arc<[AggregateFnRef]> {
        &self.data().aggregate_fns
    }

    /// The dtype of the column the zones summarise.
    pub fn column_dtype(&self) -> &DType {
        &self.data().column_dtype
    }

    /// Whether this plan tells which rows' zones may hold a match, rather than producing data.
    pub fn is_pruning(&self) -> bool {
        matches!(self.data().kind, Kind::Prune { .. })
    }

    /// The predicate over the column this plan evaluates, when its data is a conjunct.
    pub fn predicate(&self) -> Option<&BoundExpression> {
        match &self.data().kind {
            Kind::Data { predicate } => predicate.as_ref(),
            Kind::Prune { .. } => None,
        }
    }

    /// The predicate a pruning plan proves zones against.
    pub(crate) fn pruning_predicate(&self) -> Option<&BoundExpression> {
        match &self.data().kind {
            Kind::Data { .. } => None,
            Kind::Prune { predicate } => Some(predicate),
        }
    }

    /// The plan producing the data, unless this is a pruning plan.
    pub fn data_plan(&self) -> VortexResult<Option<PlanRef>> {
        if self.is_pruning() {
            return Ok(None);
        }
        Ok(Some(self.child_required(DATA)?))
    }

    /// The plan producing the zone table, one row per zone.
    pub fn zones_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(if self.is_pruning() { 0 } else { ZONES })
    }

    pub(crate) fn cache(&self) -> &Arc<ZoneCache> {
        &self.data().cache
    }

    /// A plan telling, for every row, whether the row's zone may hold a row passing this plan's
    /// predicate, or `None` when there is no predicate or no statistic proves it false.
    ///
    /// The plan reads the zone table once, through the shared cache, and no data.
    pub(crate) fn pruning_plan(&self, session: &VortexSession) -> VortexResult<Option<PlanRef>> {
        let Some(predicate) = self.predicate() else {
            return Ok(None);
        };
        if self.cache().proof(predicate, session).is_none() {
            return Ok(None);
        }
        let mut data = self.data().clone();
        data.kind = Kind::Prune {
            predicate: predicate.clone(),
        };
        Ok(Some(
            PlanParts {
                vtable: Zoned,
                dtype: DType::Bool(Nullability::NonNullable),
                row_count: self.row_count(),
                children: vec![self.zones_plan()?].into(),
                data,
            }
            .into_typed()
            .into_plan(),
        ))
    }

    /// This plan over `data` in place of its data child, with `predicate` as the conjunct the
    /// new data evaluates over the column.
    fn with_data(&self, data: PlanRef, predicate: Option<BoundExpression>) -> VortexResult<Self> {
        let mut plan_data = self.data().clone();
        plan_data.kind = Kind::Data { predicate };
        Ok(PlanParts {
            vtable: Zoned,
            dtype: data.dtype().clone(),
            row_count: self.row_count(),
            children: vec![data, self.zones_plan()?].into(),
            data: plan_data,
        }
        .into_typed())
    }
}

impl PlanVTable for Zoned {
    type PlanData = ZonedData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.zoned");
        *ID
    }

    fn fmt(plan: &Plan<Self>, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &plan.data().kind {
            Kind::Data {
                predicate: Some(predicate),
            } => write!(formatter, " predicate={predicate}"),
            Kind::Data { predicate: None } => Ok(()),
            Kind::Prune { predicate } => write!(formatter, " prune={predicate}"),
        }
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The aggregates and the predicate are not serializable on their own.
        None
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if plan.is_pruning() {
            return check_child_count("Zoned pruning", children, 1);
        }
        check_child_count("Zoned", children, 2)?;
        let data = children
            .get(DATA)?
            .ok_or_else(|| vortex_err!("Zoned data child is absent"))?;
        if data.dtype() != plan.dtype() || data.row_count() != plan.row_count() {
            vortex_bail!("Zoned data child does not produce the plan's rows");
        }
        Ok(())
    }

    fn child_name(plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match (plan.is_pruning(), index) {
            (true, 0) | (false, ZONES) => Cow::Borrowed("zones"),
            (false, DATA) => Cow::Borrowed("data"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        if plan.is_pruning() {
            return Ok(Box::new(ZonePruneNode::try_new(
                plan.clone(),
                Selection::try_new(rows, mask)?,
                ctx.session().clone(),
            )?));
        }
        plan.child_required(DATA)?.exec(rows, mask, ctx)
    }
}

/// Pushes an expression over a [`Zoned`] plan into its data, keeping a boolean one as the
/// plan's predicate so a query can prune zones with it.
///
/// Only an expression over the column itself is pushed: once one has been, the plan no longer
/// produces the column the zones summarise, and a further expression stays above it.
#[derive(Debug)]
pub(crate) struct ExpressionZonedRule;

impl PlanParentReduceRule<Zoned> for ExpressionZonedRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Zoned>,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        if child.is_pruning() || child.dtype() != child.column_dtype() {
            return Ok(None);
        }
        let Some(data) = child.data_plan()? else {
            return Ok(None);
        };
        let expression = parent.expression();
        let predicate = expression.dtype().is_boolean().then(|| expression.clone());
        let data = EvalPlan::try_new(expression.clone(), data)?.into_plan();
        Ok(Some(child.with_data(data, predicate)?.into_plan()))
    }
}
