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
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::layouts::zoned::legacy_stats_to_aggregates;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::Zoned;
use crate::plan::physical::ZonedPlan;
use crate::plan::physical::check_child_count;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::optimizer::PlanReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;
use crate::plan::physical::pipeline::Reach;
use crate::segments::SegmentId;

const DATA: usize = 0;
const ZONES: usize = 1;

/// Data alongside a legacy `vortex.stats` zone table, whose fields are named by [`Stat`] rather
/// than by aggregate function.
///
/// The optimizer rewrites it into a [`Zoned`] plan over the zone table converted to the aggregate
/// schema, or into its data alone when a [`Zoned`] plan already summarises the same data. Compiled
/// unoptimized, it reads its data and prunes nothing.
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

impl LegacyStatsPlan {
    pub(crate) fn from_children(
        dtype: DType,
        row_count: u64,
        children: PlanChildren,
        zone_len: u64,
        stats: Arc<[Stat]>,
    ) -> Self {
        PlanParts {
            vtable: LegacyStats,
            dtype,
            row_count,
            children,
            data: LegacyStatsData { zone_len, stats },
        }
        .into_typed()
    }

    /// Creates a legacy stats plan over `data` summarised by `zones` of `zone_len` rows each,
    /// whose fields hold `stats`.
    pub fn new(data: PlanRef, zones: PlanRef, zone_len: u64, stats: Arc<[Stat]>) -> Self {
        let dtype = data.dtype().clone();
        let row_count = data.row_count();
        Self::from_children(dtype, row_count, vec![data, zones].into(), zone_len, stats)
    }

    /// Returns the number of rows each zone summarises.
    pub fn zone_len(&self) -> u64 {
        self.data().zone_len
    }

    /// Returns the stats the zone table stores.
    pub fn stats(&self) -> &Arc<[Stat]> {
        &self.data().stats
    }

    /// Returns the plan producing the summarised data.
    pub fn data_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(DATA)
    }

    /// Returns the plan producing the legacy zone table.
    pub fn zones_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(ZONES)
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

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("LegacyStats", children, 2)?;
        let data = children
            .get(DATA)?
            .ok_or_else(|| vortex_err!(AssertionFailed: "LegacyStats data child is absent"))?;
        if data.dtype() != plan.dtype() || data.row_count() != plan.row_count() {
            vortex_bail!(MismatchedTypes: "LegacyStats data child shape does not match the plan output");
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            DATA => Cow::Borrowed("data"),
            ZONES => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        compiler.compile(&plan.data_plan()?, rows, mask)
    }

    fn reach(
        plan: &Plan<Self>,
        rows: Range<u64>,
        at: &Reach,
        visit: &mut dyn FnMut(SegmentId, Range<u64>),
    ) -> VortexResult<()> {
        plan.data_plan()?.reach(rows, at, visit)
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
    fn reduce(&self, plan: &Plan<LegacyStats>) -> VortexResult<Option<PlanRef>> {
        let data = plan.data_plan()?;
        if plan.zone_len() == 0 || data.is::<Zoned>() {
            return Ok(Some(data));
        }
        let zones = plan.zones_plan()?;
        let (aggregate_fns, conversion) =
            legacy_stats_to_aggregates(plan.dtype(), zones.dtype(), plan.stats());
        if aggregate_fns.is_empty() {
            return Ok(Some(data));
        }
        let zones = EvalPlan::try_new(conversion.bind(zones.dtype())?, zones)?.into_plan();
        let zoned = ZonedPlan::new(data, zones, plan.zone_len(), aggregate_fns);
        Ok(Some(zoned.into_plan()))
    }
}

/// Drops legacy stats beneath a [`Zoned`] plan: its aggregate zone map already summarises the
/// same rows.
#[derive(Debug)]
pub(crate) struct ZonedLegacyStatsRule;

impl PlanParentReduceRule<LegacyStats> for ZonedLegacyStatsRule {
    type Parent = Zoned;

    fn reduce_parent(
        &self,
        child: &LegacyStatsPlan,
        parent: &ZonedPlan,
        child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        if child_idx != DATA || parent.is_pruning() {
            return Ok(None);
        }
        let children = vec![child.data_plan()?, parent.zones_plan()?];
        Ok(Some(parent.clone().into_plan().with_children(children)?))
    }
}
