// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::ZonedPlan;
use crate::plan::exec::Event;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;
use crate::plan::exec::piece::join;
use crate::plan::plans::PrunedZones;

/// Evaluates a pruning proof over a column's zone table and returns, for every selected row,
/// whether the proof holds for the row's zone.
///
/// The zone table is read once per plan and kept on it with the per-zone result, so every later
/// execution only expands the result to its rows and issues no reads. A proof with dynamic
/// comparisons is proven again over the kept zones when one of them changes.
pub(crate) struct ZonePruneNode {
    plan: ZonedPlan,
    selection: Selection,
    session: VortexSession,
    started: bool,
    zones: Vec<Piece>,
}

impl ZonePruneNode {
    pub(crate) fn new(plan: ZonedPlan, selection: Selection, session: VortexSession) -> Self {
        Self {
            plan,
            selection,
            session,
            started: false,
            zones: Vec::new(),
        }
    }

    fn prune(&mut self) -> VortexResult<Mask> {
        prune_zones(&self.plan, &self.session, std::mem::take(&mut self.zones))
    }

    fn pruned_zones(&self) -> VortexResult<&Arc<PrunedZones>> {
        pruned_zones(&self.plan)
    }

    fn expand(&self, pruned: &Mask) -> VortexResult<Piece> {
        expand_zones(&self.plan, &self.selection, pruned)
    }
}

/// The per-zone results kept on a pruning `plan`.
pub(crate) fn pruned_zones(plan: &ZonedPlan) -> VortexResult<&Arc<PrunedZones>> {
    plan.pruned_zones()
        .ok_or_else(|| vortex_err!("ZonePruneNode needs a pruning plan"))
}

/// Builds the zone map from the pieces of the zone table and proves the expression over it, one
/// value per zone, keeping both on the plan.
pub(crate) fn prune_zones(
    plan: &ZonedPlan,
    session: &VortexSession,
    mut zones: Vec<Piece>,
) -> VortexResult<Mask> {
    let zones_plan = plan.zones_plan()?;
    zones.sort_by_key(|piece| piece.rows.start);
    let zones = zones.into_iter().map(|piece| piece.array).collect();
    let table = join(zones_plan.dtype(), zones)?;
    let table = table.execute::<StructArray>(&mut session.create_execution_ctx())?;
    let (Some(expression), Some(column_dtype)) =
        (plan.pruning_expression(), plan.pruning_column_dtype())
    else {
        vortex_bail!("ZonePruneNode needs a pruning plan");
    };
    let zone_map = match plan.legacy_stats() {
        Some(stats) => ZoneMap::try_new_legacy(
            column_dtype.clone(),
            table,
            Arc::clone(stats),
            plan.zone_len(),
            plan.row_count(),
        )?,
        None => ZoneMap::try_new(
            column_dtype.clone(),
            table,
            Arc::clone(plan.aggregate_fns()),
            plan.zone_len(),
            plan.row_count(),
        )?,
    };
    pruned_zones(plan)?.init(zone_map, expression, session)
}

/// Expands the per-zone result to one value per selected row.
///
/// Every row of a zone shares the zone's value, so the rows are filled a zone at a time and the
/// selection is applied once at the end.
pub(crate) fn expand_zones(
    plan: &ZonedPlan,
    selection: &Selection,
    pruned: &Mask,
) -> VortexResult<Piece> {
    let rows = selection.rows().clone();
    let zone_len = plan.zone_len();
    let start_zone = usize::try_from(rows.start / zone_len)?;
    let end_zone = usize::try_from(rows.end.div_ceil(zone_len))?;
    let selected_zones = pruned.slice(start_zone..end_zone);
    if selected_zones.all_true() || selected_zones.all_false() {
        let scalar = Scalar::from(selected_zones.all_true()).cast(plan.dtype())?;
        return Ok(Piece {
            rows,
            array: ConstantArray::new(scalar, selection.mask().true_count()).into_array(),
        });
    }
    let mut bits = BitBufferMut::with_capacity(selection.mask().len());
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
    let mut array = BoolArray::try_new(bits.freeze(), validity)?.into_array();
    let mask = selection.mask();
    if !mask.all_true() {
        array = array.filter(mask.clone())?;
    }
    Ok(Piece { rows, array })
}

impl ExecNode for ZonePruneNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            if self.selection.mask().all_false() {
                cx.emit(empty_piece(
                    self.plan.dtype(),
                    self.selection.rows().clone(),
                ));
                return Ok(NodeState::Done);
            }
            let expression = self
                .plan
                .pruning_expression()
                .ok_or_else(|| vortex_err!("ZonePruneNode needs a pruning plan"))?;
            if let Some(pruned) = self.pruned_zones()?.get(expression, &self.session)? {
                cx.emit(self.expand(&pruned)?);
                return Ok(NodeState::Done);
            }
            let zones = self.plan.zones_plan()?;
            let count = usize::try_from(zones.row_count())?;
            cx.spawn(0, zones, 0..count as u64, Mask::new_true(count));
        }
        for event in cx.events() {
            match event {
                Event::Piece(_, piece) => self.zones.push(piece),
                Event::Closed(_) => {
                    let pruned = self.prune()?;
                    cx.emit(self.expand(&pruned)?);
                    return Ok(NodeState::Done);
                }
                event => return Err(event.unexpected("ZonePrune")),
            }
        }
        Ok(NodeState::Wait)
    }
}
