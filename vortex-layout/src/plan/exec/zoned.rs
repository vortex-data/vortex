// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::ZonedPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;
use crate::plan::plans::Proof;

const ZONES: usize = 0;

/// Tells, for every row of its range, whether the row's zone may hold a row passing a
/// predicate, from the zone table alone.
///
/// The output is dense over the range, whatever the selection, so a query can intersect it with
/// its mask. The zone table is read once per zoned plan and kept on it with the zones each
/// predicate prunes, so most executions issue no read and emit one array at start, a constant
/// when the range's zones are all kept or all pruned.
pub(crate) struct ZonePruneNode {
    plan: ZonedPlan,
    selection: Selection,
    session: VortexSession,
    proof: Arc<Proof>,
}

impl ZonePruneNode {
    pub(crate) fn try_new(
        plan: ZonedPlan,
        selection: Selection,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let predicate = plan
            .pruning_predicate()
            .ok_or_else(|| vortex_err!("ZonePrune needs a pruning plan"))?;
        let proof = plan
            .cache()
            .proof(predicate, &session)
            .ok_or_else(|| vortex_err!("ZonePrune predicate {predicate} is not provable"))?;
        Ok(Self {
            plan,
            selection,
            session,
            proof,
        })
    }

    /// One value per row of the range: whether the row's zone is kept.
    fn expand(&self, zone_map: &ZoneMap) -> VortexResult<ArrayRef> {
        let pruned = self.proof.pruned(zone_map, &self.session)?;
        let rows = self.selection.rows();
        let zone_len = self.plan.zone_len();
        let len = usize::try_from(rows.end - rows.start)?;
        let zones = pruned.slice(
            usize::try_from(rows.start / zone_len)?..usize::try_from(rows.end.div_ceil(zone_len))?,
        );
        if zones.all_false() {
            return Ok(ConstantArray::new(true, len).into_array());
        }
        if zones.all_true() {
            return Ok(ConstantArray::new(false, len).into_array());
        }
        let mut bits = BitBufferMut::with_capacity(len);
        let mut row = rows.start;
        while row < rows.end {
            let zone = row / zone_len;
            let end = ((zone + 1) * zone_len).min(rows.end);
            bits.append_n(
                !pruned.value(usize::try_from(zone)?),
                usize::try_from(end - row)?,
            );
            row = end;
        }
        Ok(BoolArray::from(bits.freeze()).into_array())
    }
}

impl ExecNode for ZonePruneNode {
    fn ready(&self) -> Ready {
        Ready::AllClosed
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.selection.mask().all_false() {
            return Ok(NodeState::Done);
        }
        if let Some(zone_map) = self.plan.cache().zone_map() {
            cx.emit(self.expand(zone_map)?);
            return Ok(NodeState::Done);
        }
        let zones = self.plan.zones_plan()?;
        let count = zones.row_count();
        cx.spawn(
            ZONES,
            zones,
            0..count,
            Mask::new_true(usize::try_from(count)?),
        );
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let zones = self.plan.zones_plan()?;
        let table = join(zones.dtype(), cx.input(ZONES).take_all())?
            .execute::<StructArray>(&mut self.session.create_execution_ctx())?;
        let zone_map = ZoneMap::try_new(
            self.plan.column_dtype().clone(),
            table,
            Arc::clone(self.plan.aggregate_fns()),
            self.plan.zone_len(),
            self.plan.row_count(),
        )?;
        let zone_map = self.plan.cache().set_zone_map(zone_map);
        cx.emit(self.expand(zone_map)?);
        Ok(NodeState::Done)
    }
}
