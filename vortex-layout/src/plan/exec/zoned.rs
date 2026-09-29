// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::ZonedPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
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
    started: bool,
    zones: Vec<Piece>,
}

impl ZonePruneNode {
    pub(crate) fn new(plan: ZonedPlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
            zones: Vec::new(),
        }
    }

    /// Builds the zone map from the joined zone table and proves the expression over it, one
    /// value per zone, keeping both on the plan.
    fn prune(&mut self, cx: &StepCx<'_>) -> VortexResult<Mask> {
        let zones_plan = self.plan.zones_plan()?;
        self.zones.sort_by_key(|piece| piece.rows.start);
        let zones = std::mem::take(&mut self.zones)
            .into_iter()
            .map(|piece| piece.array)
            .collect();
        let table = join(zones_plan.dtype(), zones)?;
        let table = table.execute::<StructArray>(&mut cx.session().create_execution_ctx())?;
        let (Some(expression), Some(column_dtype)) = (
            self.plan.pruning_expression(),
            self.plan.pruning_column_dtype(),
        ) else {
            vortex_bail!("ZonePruneNode needs a pruning plan");
        };
        let zone_map = ZoneMap::try_new(
            column_dtype.clone(),
            table,
            Arc::clone(self.plan.aggregate_fns()),
            self.plan.zone_len(),
            self.plan.row_count(),
        )?;
        self.pruned_zones()?
            .init(zone_map, expression, cx.session())
    }

    fn pruned_zones(&self) -> VortexResult<&Arc<PrunedZones>> {
        self.plan
            .pruned_zones()
            .ok_or_else(|| vortex_err!("ZonePruneNode needs a pruning plan"))
    }

    /// Expands the per-zone result to one value per selected row.
    ///
    /// Every row of a zone shares the zone's value, so the rows are filled a zone at a time and
    /// the selection is applied once at the end.
    fn expand(&self, pruned: &Mask) -> VortexResult<Piece> {
        let rows = self.selection.rows().clone();
        let zone_len = self.plan.zone_len();
        let mut bits = BitBufferMut::with_capacity(self.selection.mask().len());
        let mut row = rows.start;
        while row < rows.end {
            let zone =
                usize::try_from(row / zone_len).vortex_expect("zone index must fit in usize");
            let zone_end = ((row / zone_len) + 1) * zone_len;
            let end = zone_end.min(rows.end);
            bits.append_n(
                pruned.value(zone),
                usize::try_from(end - row).vortex_expect("zone rows must fit in usize"),
            );
            row = end;
        }
        let validity = if self.plan.dtype().is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        let mut array = BoolArray::try_new(bits.freeze(), validity)?.into_array();
        let mask = self.selection.mask();
        if !mask.all_true() {
            array = array.filter(mask.clone())?;
        }
        Ok(Piece { rows, array })
    }
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
                cx.close();
                return Ok(NodeState::Done);
            }
            let expression = self
                .plan
                .pruning_expression()
                .ok_or_else(|| vortex_err!("ZonePruneNode needs a pruning plan"))?;
            if let Some(pruned) = self.pruned_zones()?.get(expression, cx.session())? {
                cx.emit(self.expand(&pruned)?);
                cx.close();
                return Ok(NodeState::Done);
            }
            let zones = self.plan.zones_plan()?;
            let count = usize::try_from(zones.row_count())?;
            cx.spawn(0, zones, 0..count as u64, Mask::new_true(count));
        }
        for (_, input) in cx.take_inputs() {
            match input {
                Input::Piece(piece) => self.zones.push(piece),
                Input::Closed => {
                    let pruned = self.prune(cx)?;
                    cx.emit(self.expand(&pruned)?);
                    cx.close();
                    return Ok(NodeState::Done);
                }
            }
        }
        Ok(NodeState::Waiting)
    }
}
