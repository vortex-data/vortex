// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::physical::Eval;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::check_child_count;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::operators::UNBOUNDED;
use crate::plan::physical::operators::WholeSource;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::optimizer::PlanReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

const DATA: usize = 0;
const ZONES: usize = 1;

#[derive(Clone, Debug)]
struct ZonedPruningState {
    expression: BoundExpression,
    /// The dtype of the column the zones summarise, which the expression is bound to.
    column_dtype: DType,
    /// Which zones the expression proves can be pruned, shared by every execution of this plan.
    pruned_zones: Arc<PrunedZones>,
}

/// Which zones a pruning expression proves can be pruned, shared by every execution of a plan.
///
/// The zone table is read once and kept. An expression with dynamic comparisons, such as a top-N
/// bound an engine tightens during the scan, is proven again over the kept zones whenever one of
/// them changes, as the default scan's zoned reader does.
pub(crate) struct PrunedZones {
    updates: Option<DynamicExprUpdates>,
    /// The zones, and the pruned zones as of an update version.
    state: RwLock<Option<(ZoneMap, u64, Mask)>>,
}

impl PrunedZones {
    fn new(expression: &BoundExpression) -> Self {
        Self {
            updates: DynamicExprUpdates::new(expression),
            state: RwLock::new(None),
        }
    }

    fn version(&self) -> u64 {
        self.updates.as_ref().map_or(0, DynamicExprUpdates::version)
    }

    /// The pruned zones, proven again if a dynamic comparison changed, or `None` before any
    /// execution read the zones.
    pub(crate) fn get(
        &self,
        expression: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<Mask>> {
        let version = self.version();
        {
            let state = self.state.read();
            match &*state {
                None => return Ok(None),
                Some((_, proven, pruned)) if *proven >= version => return Ok(Some(pruned.clone())),
                Some(_) => {}
            }
        }
        let mut state = self.state.write();
        let Some((zones, proven, pruned)) = state.as_mut() else {
            return Ok(None);
        };
        if *proven < version {
            *pruned = zones.prune(expression, session)?;
            *proven = version;
        }
        Ok(Some(pruned.clone()))
    }

    /// Proves `expression` over `zones` and keeps both, unless another execution did first.
    pub(crate) fn init(
        &self,
        zones: ZoneMap,
        expression: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let version = self.version();
        let pruned = zones.prune(expression, session)?;
        let mut state = self.state.write();
        if state.is_none() {
            *state = Some((zones, version, pruned.clone()));
        }
        Ok(pruned)
    }
}

impl fmt::Debug for PrunedZones {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrunedZones")
            .field("dynamic", &self.updates.is_some())
            .field("read", &self.state.read().is_some())
            .finish()
    }
}

/// Zoned-plan-specific data.
#[derive(Clone, Debug)]
pub struct ZonedData {
    zone_len: u64,
    /// The aggregate functions whose results the zone table stores, in field order.
    aggregate_fns: Arc<[AggregateFnRef]>,
    pruning: Option<ZonedPruningState>,
}

/// Reads data alongside the zone statistics summarising it.
///
/// Its zone table has the aggregate schema of `vortex.zoned` layouts; legacy `vortex.stats`
/// layouts reach it through [`LegacyStats`](crate::plan::physical::LegacyStats), which converts
/// their tables. An expression containing abstract statistic functions can rewrite it into a
/// pruning plan that retains only the zone-statistics child; a plan that reads the data is
/// optimized to its data child alone.
#[derive(Clone, Debug)]
pub struct Zoned;

/// A plan that pairs data with its zone statistics or represents a zone-backed pruning proof.
pub type ZonedPlan = Plan<Zoned>;

impl ZonedPlan {
    pub(crate) fn from_children(
        dtype: DType,
        row_count: u64,
        children: PlanChildren,
        zone_len: u64,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> Self {
        PlanParts {
            vtable: Zoned,
            dtype,
            row_count,
            children,
            data: ZonedData {
                zone_len,
                aggregate_fns,
                pruning: None,
            },
        }
        .into_typed()
    }

    /// Creates a zoned plan over `data` summarised by `zones` of `zone_len` rows each, whose
    /// fields hold the results of `aggregate_fns`.
    pub fn new(
        data: PlanRef,
        zones: PlanRef,
        zone_len: u64,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> Self {
        let dtype = data.dtype().clone();
        let row_count = data.row_count();
        Self::from_children(
            dtype,
            row_count,
            vec![data, zones].into(),
            zone_len,
            aggregate_fns,
        )
    }

    /// Returns the number of rows each zone summarises.
    pub fn zone_len(&self) -> u64 {
        self.data().zone_len
    }

    /// Returns the aggregate functions whose results the zone table stores.
    pub fn aggregate_fns(&self) -> &Arc<[AggregateFnRef]> {
        &self.data().aggregate_fns
    }

    /// Returns the dtype of the column the zones summarise, when this is a pruning plan.
    pub fn pruning_column_dtype(&self) -> Option<&DType> {
        self.data()
            .pruning
            .as_ref()
            .map(|state| &state.column_dtype)
    }

    /// Returns the cache of pruned zones shared by every execution of this pruning plan.
    pub(crate) fn pruned_zones(&self) -> Option<&Arc<PrunedZones>> {
        self.data()
            .pruning
            .as_ref()
            .map(|state| &state.pruned_zones)
    }

    /// Returns the plan producing the summarised data, unless this is a pruning plan.
    pub fn data_plan(&self) -> VortexResult<Option<PlanRef>> {
        if self.is_pruning() {
            return Ok(None);
        }
        self.child(DATA)
    }

    /// Returns the plan producing the zone statistics.
    pub fn zones_plan(&self) -> VortexResult<PlanRef> {
        let index = if self.is_pruning() { 0 } else { ZONES };
        self.child_required(index)
    }

    /// Returns whether this plan represents a zone-backed pruning proof.
    pub fn is_pruning(&self) -> bool {
        self.data().pruning.is_some()
    }

    /// Returns the abstract pruning proof carried by this plan, when present.
    pub fn pruning_expression(&self) -> Option<&BoundExpression> {
        self.data().pruning.as_ref().map(|state| &state.expression)
    }

    fn with_pruning(&self, expression: BoundExpression) -> VortexResult<Option<Self>> {
        if self.data().zone_len == 0 || self.is_pruning() {
            return Ok(None);
        }
        let mut data = self.data().clone();
        data.pruning = Some(ZonedPruningState {
            expression: expression.clone(),
            column_dtype: self.dtype().clone(),
            pruned_zones: Arc::new(PrunedZones::new(&expression)),
        });
        Ok(Some(
            PlanParts {
                vtable: Zoned,
                dtype: expression.dtype().clone(),
                row_count: self.row_count(),
                children: vec![self.zones_plan()?].into(),
                data,
            }
            .into_typed(),
        ))
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
        if let Some(expression) = plan.pruning_expression() {
            write!(formatter, " prune={expression}")?;
        }
        Ok(())
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if plan.is_pruning() {
            check_child_count("Zoned pruning", children, 1)?;
            // New zones may prove different zones, so the proof of the old ones is not kept.
            if let Some(state) = &mut data.pruning {
                state.pruned_zones = Arc::new(PrunedZones::new(&state.expression));
            }
            return Ok(());
        }

        check_child_count("Zoned", children, 2)?;
        let data = children
            .get(DATA)?
            .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?;
        if data.dtype() != plan.dtype() || data.row_count() != plan.row_count() {
            vortex_error::vortex_bail!("Zoned data child shape does not match the plan output");
        }
        Ok(())
    }

    fn child_name(plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if plan.is_pruning() {
            return if index == 0 {
                Cow::Borrowed("zones")
            } else {
                Cow::Owned(format!("child[{index}]"))
            };
        }
        match index {
            DATA => Cow::Borrowed("data"),
            ZONES => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    /// A data plan compiles as its data child. A pruning plan whose zones an earlier execution
    /// proved emits the expanded result; otherwise it collects the whole zone table, proves the
    /// expression over it, keeps the result, and emits it expanded to the selected rows.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if !plan.is_pruning() {
            let Some(data) = plan.data_plan()? else {
                vortex_bail!("Zoned plan has no data child");
            };
            return compiler.compile(&data, rows, mask);
        }
        if mask.all_false() {
            return Ok(None);
        }
        let expression = plan
            .pruning_expression()
            .ok_or_else(|| vortex_err!("Zoned pruning plan has no expression"))?;
        if let Some(pruned) = pruned_zones(plan)?.get(expression, compiler.session())? {
            return Ok(Some(Chain::new(OnceSource::new(expand_zones(
                plan, &rows, mask, &pruned,
            )?))));
        }
        let zones = plan.zones_plan()?;
        let count = usize::try_from(zones.row_count())?;
        let Some(chain) = compiler.compile(&zones, 0..zones.row_count(), &Mask::new_true(count))?
        else {
            vortex_bail!("Zoned pruning plan has an empty zone table");
        };
        let zoned = plan.clone();
        let mask = mask.clone();
        let source = WholeSource::new(zones.dtype().clone(), move |table, cx| {
            let pruned = prune_zones(&zoned, cx.session(), table)?;
            expand_zones(&zoned, &rows, &mask, &pruned)
        });
        Ok(Some(compiler.join_with(vec![chain], |_| UNBOUNDED, source)))
    }
}

/// The per-zone results kept on a pruning `plan`.
fn pruned_zones(plan: &ZonedPlan) -> VortexResult<&Arc<PrunedZones>> {
    plan.pruned_zones()
        .ok_or_else(|| vortex_err!("Zone pruning needs a pruning plan"))
}

/// Builds the zone map from the zone table and proves the expression over it, one value per
/// zone, keeping both on the plan.
fn prune_zones(plan: &ZonedPlan, session: &VortexSession, table: ArrayRef) -> VortexResult<Mask> {
    let table = table.execute::<StructArray>(&mut session.create_execution_ctx())?;
    let (Some(expression), Some(column_dtype)) =
        (plan.pruning_expression(), plan.pruning_column_dtype())
    else {
        vortex_bail!("Zone pruning needs a pruning plan");
    };
    let zone_map = ZoneMap::try_new(
        column_dtype.clone(),
        table,
        Arc::clone(plan.aggregate_fns()),
        plan.zone_len(),
        plan.row_count(),
    )?;
    pruned_zones(plan)?.init(zone_map, expression, session)
}

/// Expands the per-zone result to one value per row of `rows` that `mask` selects.
///
/// Every row of a zone shares the zone's value, so the rows are filled a zone at a time and the
/// selection is applied once at the end.
fn expand_zones(
    plan: &ZonedPlan,
    rows: &Range<u64>,
    mask: &Mask,
    pruned: &Mask,
) -> VortexResult<ArrayRef> {
    let zone_len = plan.zone_len();
    let start_zone = usize::try_from(rows.start / zone_len)?;
    let end_zone = usize::try_from(rows.end.div_ceil(zone_len))?;
    let selected_zones = pruned.slice(start_zone..end_zone);
    if selected_zones.all_true() || selected_zones.all_false() {
        let scalar = Scalar::from(selected_zones.all_true()).cast(plan.dtype())?;
        return Ok(ConstantArray::new(scalar, mask.true_count()).into_array());
    }
    let mut bits = BitBufferMut::with_capacity(mask.len());
    let mut row = rows.start;
    while row < rows.end {
        let zone = usize::try_from(row / zone_len).vortex_expect("zone index must fit in usize");
        let zone_end = ((row / zone_len) + 1) * zone_len;
        let end = zone_end.min(rows.end);
        bits.append_n(
            pruned.value(zone),
            usize::try_from(end - row).vortex_expect("zone rows must fit in usize"),
        );
        row = end;
    }
    let validity = if plan.dtype().is_nullable() {
        Validity::AllValid
    } else {
        Validity::NonNullable
    };
    let array = BoolArray::try_new(bits.freeze(), validity)?.into_array();
    if mask.all_true() {
        return Ok(array);
    }
    array.filter(mask.clone())
}

/// Rewrites an abstract statistic expression over a zoned plan into its pruning state.
#[derive(Debug)]
pub(crate) struct ExpressionZonedRule;

impl PlanParentReduceRule<Zoned> for ExpressionZonedRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &ZonedPlan,
        parent: &Plan<Eval>,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        if !is_zone_pruning(parent)? {
            return Ok(None);
        }
        Ok(child
            .with_pruning(parent.expression().clone())?
            .map(Plan::into_plan))
    }
}

/// Whether `eval` is a boolean expression over abstract statistics alone, which a zone table can
/// prune without reading the data it summarises.
pub(crate) fn is_zone_pruning(eval: &Plan<Eval>) -> VortexResult<bool> {
    if !eval.dtype().is_boolean() {
        return Ok(false);
    }
    let mut contains_stat = false;
    let mut contains_root = false;
    eval.expression().clone().transform_down(|expression| {
        if expression
            .as_scalar()
            .is_some_and(|scalar_fn| scalar_fn.is::<StatFn>())
        {
            contains_stat = true;
            return Ok(Transformed {
                value: expression,
                order: TraversalOrder::Skip,
                changed: false,
            });
        }
        contains_root |= expression.is_root();
        Ok(Transformed::no(expression))
    })?;
    Ok(contains_stat && !contains_root)
}

/// Replaces a zoned data plan by its data: a scan reads the data through it and never its zone
/// table. Pruning plans are derived from the filter over the lowered plan before this runs, so
/// an optimized plan carries a zone table only where a pruning plan reads it.
#[derive(Debug)]
pub(crate) struct ZonedDataRule;

impl PlanReduceRule<Zoned> for ZonedDataRule {
    fn reduce(
        &self,
        plan: &Plan<Zoned>,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        plan.data_plan()
    }
}
