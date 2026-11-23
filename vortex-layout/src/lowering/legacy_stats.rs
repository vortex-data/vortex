// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
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
use vortex_scan::plan::Zoned;
use vortex_scan::plan::ZonedPlan;
use vortex_scan::plan::is_zone_pruning;
use vortex_scan::plan::optimizer::PlanParentReduceRule;
use vortex_scan::plan::optimizer::PlanParentReduceRuleAdapter;
use vortex_scan::plan::optimizer::PlanParentRuleSet;
use vortex_scan::plan::optimizer::PlanReduceRule;
use vortex_scan::plan::optimizer::PlanReduceRuleAdapter;
use vortex_scan::plan::optimizer::PlanRuleSet;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::zoned::legacy_stats_to_aggregates;

const DATA: usize = 0;
const ZONES: usize = 1;

/// Data alongside a legacy `vortex.stats` zone table, whose fields are named by [`Stat`] rather
/// than by aggregate function.
///
/// The optimizer rewrites it into a [`Zoned`] plan over the zone table converted to the aggregate
/// schema, or into its data alone when a [`Zoned`] plan already summarises the same data; a plan
/// that reads the data is then optimized to the data alone. Compiled unoptimized, it reads its
/// data and prunes nothing.
#[derive(Clone, Debug)]
pub struct LegacyStats;

/// Legacy-stats-plan-specific data.
#[derive(Clone, Debug)]
pub struct LegacyStatsData {
    zone_len: u64,
    /// The stats the zone table stores.
    stats: Arc<[Stat]>,
}

/// A plan that pairs data with a legacy stats zone table.
pub type LegacyStatsPlan = Plan<LegacyStats>;

impl LegacyStats {
    pub(crate) fn from_slots(
        dtype: DType,
        row_count: u64,
        slots: PlanSlots,
        zone_len: u64,
        stats: Arc<[Stat]>,
    ) -> LegacyStatsPlan {
        PlanParts {
            vtable: LegacyStats,
            dtype,
            row_count,
            slots,
            data: LegacyStatsData { zone_len, stats },
        }
        .into_typed()
    }

    /// Creates a legacy stats plan over `data` summarised by `zones` of `zone_len` rows each,
    /// whose fields hold `stats`.
    pub fn new_plan(
        data: PlanRef,
        zones: PlanRef,
        zone_len: u64,
        stats: Arc<[Stat]>,
    ) -> LegacyStatsPlan {
        let dtype = data.dtype().clone();
        let row_count = data.row_count();
        Self::from_slots(dtype, row_count, [data, zones].into(), zone_len, stats)
    }

    pub(super) fn data_plan(plan: &LegacyStatsPlan) -> VortexResult<PlanRef> {
        plan.slot_required(DATA)
    }
}

impl PlanVTable for LegacyStats {
    type PlanData = LegacyStatsData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.legacy_stats");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("LegacyStats", 2)?;
        let data = slots.required(DATA)?;
        if data.dtype() != plan.dtype() || data.row_count() != plan.row_count() {
            vortex_bail!("LegacyStats data child shape does not match the plan output");
        }
        Ok(())
    }

    fn slot_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            DATA => Cow::Borrowed("data"),
            ZONES => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("slot[{index}]")),
        }
    }

    fn reduce(plan: &Plan<Self>, session: &VortexSession) -> VortexResult<Option<PlanRef>> {
        static RULES: PlanRuleSet =
            PlanRuleSet::new(&[&PlanReduceRuleAdapter::new(LegacyStatsRule)]);
        RULES.evaluate(&plan.clone().into_plan(), session)
    }

    fn reduce_parent(
        plan: &Plan<Self>,
        parent: &PlanRef,
        slot: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        static RULES: PlanParentRuleSet =
            PlanParentRuleSet::new(&[&PlanParentReduceRuleAdapter::new(ExpressionLegacyStatsRule)]);
        RULES.evaluate(&plan.clone().into_plan(), parent, slot, session)
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        compiler.compile(&LegacyStats::data_plan(plan)?, rows, mask)
    }
}

/// Rewrites a legacy stats plan into a [`Zoned`] plan over its zone table converted to the
/// aggregate schema.
///
/// A plan whose data is already [`Zoned`] keeps only that data: legacy stats are used only when no
/// aggregate zone map summarises the same rows. A zero zone length, as older files may record, or
/// a table with no stat an aggregate can stand for, leaves only the data too.
#[derive(Debug)]
pub(crate) struct LegacyStatsRule;

impl PlanReduceRule<LegacyStats> for LegacyStatsRule {
    fn reduce(
        &self,
        plan: &Plan<LegacyStats>,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let data = LegacyStats::data_plan(plan)?;
        if plan.data().zone_len == 0 || data.is::<Zoned>() {
            return Ok(Some(data));
        }
        let zones = plan.slot_required(ZONES)?;
        let (aggregate_fns, conversion) =
            legacy_stats_to_aggregates(plan.dtype(), zones.dtype(), &plan.data().stats);
        if aggregate_fns.is_empty() {
            return Ok(Some(data));
        }
        let zones = EvalPlan::try_new(conversion.bind(zones.dtype())?, zones)?.into_plan();
        let zoned = ZonedPlan::new(data, zones, plan.data().zone_len, aggregate_fns);
        Ok(Some(zoned.into_plan()))
    }
}

/// Converts the legacy stats beneath a zone pruning plan in place, so the plan meets a [`Zoned`] plan
/// before the data path optimizes the stats away.
#[derive(Debug)]
pub(crate) struct ExpressionLegacyStatsRule;

impl PlanParentReduceRule<LegacyStats> for ExpressionLegacyStatsRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &LegacyStatsPlan,
        parent: &Plan<Eval>,
        _child_idx: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        if !is_zone_pruning(parent)? {
            return Ok(None);
        }
        let Some(converted) = LegacyStatsRule.reduce(child, session)? else {
            return Ok(None);
        };
        Ok(Some(
            EvalPlan::try_new(parent.expression().clone(), converted)?.into_plan(),
        ))
    }
}
