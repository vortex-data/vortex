// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::optimizer::PlanParentReduceRule;

const DATA: usize = 0;
const ZONES: usize = 1;

type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;

#[derive(Clone)]
struct ZonedPruningState {
    expression: BoundExpression,
    zone_map: Arc<OnceLock<SharedZoneMap>>,
}

impl fmt::Debug for ZonedPruningState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ZonedPruningState")
            .field("expression", &self.expression)
            .finish_non_exhaustive()
    }
}

impl ZonedPruningState {
    fn new(expression: BoundExpression) -> Self {
        Self {
            expression,
            zone_map: Arc::new(OnceLock::new()),
        }
    }

    fn zone_map(
        &self,
        ctx: &PlanExecutionContext,
        zones: &PlanRef,
        column_dtype: &DType,
        aggregate_fns: &Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<SharedZoneMap> {
        let zone_count = zones.row_count();
        let zone_count_usize = usize::try_from(zone_count)?;
        Ok(self
            .zone_map
            .get_or_init(|| {
                let ctx = ctx.clone();
                let zones = zones.clone();
                let column_dtype = column_dtype.clone();
                let aggregate_fns = Arc::clone(aggregate_fns);
                async move {
                    let zones = zones.execute(
                        &ctx,
                        &(0..zone_count),
                        MaskFuture::new_true(zone_count_usize),
                    )?;
                    let mut execution = ctx.session().create_execution_ctx();
                    let zones = zones.await?.execute::<StructArray>(&mut execution)?;
                    // SAFETY: zoned layout construction validated that the auxiliary child was
                    // written from this column dtype and stats-table schema.
                    Ok(unsafe {
                        ZoneMap::new_unchecked(
                            column_dtype,
                            zones,
                            aggregate_fns,
                            zone_len,
                            row_count,
                        )
                    })
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone())
    }
}

/// Zoned-plan-specific data.
#[derive(Clone, Debug)]
pub struct ZonedData {
    column_dtype: DType,
    zone_len: u64,
    aggregate_fns: Arc<[AggregateFnRef]>,
    pruning: Option<ZonedPruningState>,
}

/// Reads data alongside the zone statistics summarising it.
///
/// This operator covers both `vortex.zoned` layouts and legacy `vortex.stats` layouts, which have
/// the same physical child shape. An expression containing abstract statistic functions can
/// rewrite it into a pruning plan that retains only the zone-statistics child.
#[derive(Clone, Debug)]
pub struct Zoned;

/// A plan that pairs data with its zone statistics or evaluates a zone-backed pruning proof.
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
            dtype: dtype.clone(),
            row_count,
            children,
            data: ZonedData {
                column_dtype: dtype,
                zone_len,
                aggregate_fns,
                pruning: None,
            },
        }
        .into_typed()
    }

    /// Creates a zoned plan over `data` summarised by `zones`.
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

    /// Returns whether this plan evaluates a zone-backed pruning proof.
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
        data.pruning = Some(ZonedPruningState::new(expression.clone()));
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

    fn with_data_expression(&self, expression: BoundExpression) -> VortexResult<Option<Self>> {
        let Some(data_plan) = self.data_plan()? else {
            return Ok(None);
        };
        Ok(Some(
            PlanParts {
                vtable: Zoned,
                dtype: expression.dtype().clone(),
                row_count: self.row_count(),
                children: vec![
                    EvalPlan::try_new(expression, data_plan)?.into_plan(),
                    self.zones_plan()?,
                ]
                .into(),
                data: self.data().clone(),
            }
            .into_typed(),
        ))
    }

    fn execute_pruning(
        &self,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= self.row_count(),
            "Zoned pruning row range {:?} is outside 0..{}",
            row_range,
            self.row_count()
        );
        let range_len = usize::try_from(row_range.end - row_range.start)?;
        vortex_ensure!(
            mask.len() == range_len,
            "Zoned pruning mask length mismatch"
        );

        let state = self
            .data()
            .pruning
            .clone()
            .ok_or_else(|| vortex_error::vortex_err!("Zoned pruning state is absent"))?;
        let ctx = ctx.clone();
        let zones = self.zones_plan()?;
        let column_dtype = self.data().column_dtype.clone();
        let output_dtype = self.dtype().clone();
        let aggregate_fns = Arc::clone(&self.data().aggregate_fns);
        let zone_len = self.data().zone_len;
        let row_count = self.row_count();
        let row_range = row_range.clone();

        Ok(async move {
            let input_mask = mask.await?;
            if input_mask.all_false() {
                return Ok(BoolArray::new(
                    BitBuffer::new_unset(0),
                    Validity::from(output_dtype.nullability()),
                )
                .into_array());
            }

            let zone_map = state.zone_map(
                &ctx,
                &zones,
                &column_dtype,
                &aggregate_fns,
                zone_len,
                row_count,
            )?;
            let zone_map = zone_map.await?;
            let evaluated = zone_map.evaluate(&state.expression, ctx.session())?;
            let mut execution = ctx.session().create_execution_ctx();
            let zone_validity =
                BoolArrayExt::validity(&evaluated).execute_mask(evaluated.len(), &mut execution)?;
            let zone_values = evaluated.to_bit_buffer();

            let zone_start = row_range.start / zone_len;
            let zone_end = row_range.end.div_ceil(zone_len);
            let zone_start_usize = usize::try_from(zone_start)?;
            let zone_end_usize = usize::try_from(zone_end)?;
            vortex_ensure!(
                zone_end_usize <= evaluated.len(),
                "Zoned pruning requires zones {zone_start}..{zone_end}, but only {} exist",
                evaluated.len()
            );

            let mut values = BitBufferMut::with_capacity(range_len);
            let mut validity = BitBufferMut::with_capacity(range_len);
            let relevant_values = zone_values.slice(zone_start_usize..zone_end_usize);
            let relevant_validity = zone_validity.slice(zone_start_usize..zone_end_usize);
            for (offset, (value, valid)) in relevant_values
                .iter()
                .zip(relevant_validity.iter())
                .enumerate()
            {
                let zone_index = zone_start + u64::try_from(offset)?;
                let zone_row_start = zone_index.saturating_mul(zone_len).min(row_count);
                let zone_row_end = zone_index
                    .saturating_add(1)
                    .saturating_mul(zone_len)
                    .min(row_count);
                let start = zone_row_start.max(row_range.start);
                let end = zone_row_end.min(row_range.end);
                if start < end {
                    let len = usize::try_from(end - start)?;
                    values.append_n(value, len);
                    validity.append_n(valid, len);
                }
            }
            vortex_ensure!(
                values.len() == range_len && validity.len() == range_len,
                "Expanded zone proof length does not match row range"
            );

            let validity = if output_dtype.is_nullable() {
                Validity::from(validity.freeze())
            } else {
                vortex_ensure!(
                    validity.freeze().true_count() == range_len,
                    "Non-nullable zoned proof produced null values"
                );
                Validity::NonNullable
            };
            let output = BoolArray::new(values.freeze(), validity).into_array();
            if input_mask.all_true() {
                Ok(output)
            } else {
                output.filter(input_mask)
            }
        }
        .boxed())
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
            let pruning = data
                .pruning
                .as_mut()
                .ok_or_else(|| vortex_error::vortex_err!("Zoned pruning state is absent"))?;
            pruning.zone_map = Arc::new(OnceLock::new());
            return Ok(());
        }

        check_child_count("Zoned", children, 2)?;
        let data_plan = children
            .get(DATA)?
            .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?;
        if data_plan.dtype() != plan.dtype() || data_plan.row_count() != plan.row_count() {
            vortex_error::vortex_bail!("Zoned data child shape does not match the plan output");
        }
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        if plan.is_pruning() {
            return plan.execute_pruning(ctx, row_range, mask);
        }
        plan.data_plan()?
            .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?
            .execute(ctx, row_range, mask)
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
}

/// Pushes data expressions through a zoned plan and rewrites statistic expressions into pruning.
#[derive(Debug)]
pub(crate) struct ExpressionZonedRule;

impl PlanParentReduceRule<Zoned> for ExpressionZonedRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &ZonedPlan,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let mut contains_stat = false;
        let mut contains_root = false;
        parent.expression().clone().transform_down(|expression| {
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
        if contains_stat {
            if !parent.dtype().is_boolean() || contains_root {
                return Ok(None);
            }
            return Ok(child
                .with_pruning(parent.expression().clone())?
                .map(Plan::into_plan));
        }

        Ok(child
            .with_data_expression(parent.expression().clone())?
            .map(Plan::into_plan))
    }
}
