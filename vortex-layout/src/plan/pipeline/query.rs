// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Query run as stages over one split: zone pruning, each conjunct under the rows the earlier
//! ones kept, then the projection.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use vortex_array::ArrayRef;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Reach;
use super::Shared;
use super::port::PortId;
use super::scan::Core;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::QueryPlan;
use crate::plan::Take;
use crate::plan::Zoned;
use crate::plan::ZonedPlan;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Pruning with the zones of the conjuncts from this index on.
    Pruning(usize),
    Conjuncts,
    Projecting,
    Done,
}

/// The rows a conjunct was compiled under, so its result can be folded into the mask.
enum Spawned {
    /// The selected rows: the result has one value per selected row.
    Selected,
    /// Every row of the chunks holding a selected row: the result has one value per row this
    /// mask selects.
    Chunks(Mask),
}

/// The stage running.
enum Current {
    /// Reading the zone table of conjunct `index`'s column.
    Zones(ZonedPlan),
    Conjunct(usize, Spawned),
    Projection,
}

/// One split of a query, run as stages. Each stage is compiled under the mask the stages
/// before it left, so a stage never builds, and never reads, a chunk no selected row is in.
pub(crate) struct QueryRun {
    plan: QueryPlan,
    rows: Range<u64>,
    /// The rows still selected.
    mask: Mask,
    phase: Phase,
    /// Conjuncts not yet evaluated.
    remaining: BitVec,
    current: Option<Current>,
    /// The running conjunct's result, folded to bits as its batches arrive, so no lazy array
    /// outlives its batch.
    folded: BitBufferMut,
    /// The zone table read so far.
    zones: Vec<ArrayRef>,
    /// Whether the scan has been told if this split projects.
    reported: bool,
}

impl QueryRun {
    pub(crate) fn new(plan: QueryPlan, rows: Range<u64>, mask: Mask) -> Self {
        let conjuncts = plan.conjunct_count();
        Self {
            plan,
            rows,
            mask,
            phase: Phase::Pruning(0),
            remaining: BitVec::from_elem(conjuncts, true),
            current: None,
            folded: BitBufferMut::with_capacity(0),
            zones: Vec::new(),
            reported: false,
        }
    }

    /// Records the segments every stage of every split could read, so segments read more
    /// than once are decoded once: the conjuncts' and the projection's over the split's rows,
    /// and, for each conjunct whose zones can prune it and whose zone table no earlier scan
    /// read, the zone table by every split.
    pub(crate) fn reserve(plan: &QueryPlan, core: &mut Core) -> VortexResult<()> {
        let rows = 0..plan.row_count();
        for index in 0..plan.conjunct_count() {
            if let Some(pruning) = plan.pruning(index, &core.session)? {
                let zones = pruning.as_::<Zoned>().zones_plan()?;
                let count = zones.row_count();
                core.shares.add_every_split(&zones, 0..count)?;
            }
            core.shares.add(&plan.conjunct(index)?, rows.clone())?;
        }
        core.shares.add(&plan.projection()?, rows)
    }

    /// Takes a batch of the running stage's output. Returns the batch when it is the split's
    /// output.
    pub(crate) fn accept(
        &mut self,
        batch: ArrayRef,
        core: &mut Core,
    ) -> VortexResult<Option<ArrayRef>> {
        match &self.current {
            Some(Current::Zones(_)) => {
                self.zones.push(batch);
                Ok(None)
            }
            Some(Current::Conjunct(..)) => {
                let batch = if batch.dtype().is_nullable() {
                    batch.fill_null(false)?
                } else {
                    batch
                };
                let mask = batch.execute::<Mask>(&mut core.exec)?;
                match mask.bit_buffer() {
                    AllOr::All => self.folded.append_n(true, mask.len()),
                    AllOr::None => self.folded.append_n(false, mask.len()),
                    AllOr::Some(bits) => self.folded.append_buffer(bits),
                }
                Ok(None)
            }
            Some(Current::Projection) => Ok(Some(batch)),
            None => Err(vortex_err!("Query has no stage running")),
        }
    }

    /// Folds the finished stage's result into the mask.
    pub(crate) fn finish_stage(&mut self, core: &mut Core) -> VortexResult<()> {
        match self.current.take() {
            Some(Current::Zones(pruning)) => {
                let zones = pruning.zones_plan()?;
                let table = super::ops::join(zones.dtype(), std::mem::take(&mut self.zones))?
                    .execute::<StructArray>(&mut core.exec)?;
                let zone_map = ZoneMap::try_new(
                    pruning.column_dtype().clone(),
                    table,
                    Arc::clone(pruning.aggregate_fns()),
                    pruning.zone_len(),
                    pruning.row_count(),
                )?;
                let zone_map = Arc::new(zone_map);
                core.zone_maps
                    .insert(zones.as_ptr_key(), Arc::clone(&zone_map));
                self.prune(&pruning, &zone_map, core)?;
            }
            Some(Current::Conjunct(index, spawned)) => {
                let bits = std::mem::replace(&mut self.folded, BitBufferMut::with_capacity(0));
                let result = Mask::from_buffer(bits.freeze());
                let input = self.mask.true_count();
                let mask = std::mem::replace(&mut self.mask, Mask::new_false(0));
                self.mask = match spawned {
                    Spawned::Selected => mask.intersect_by_rank(&result),
                    Spawned::Chunks(chunks) if chunks.all_true() => mask.bitand(&result),
                    Spawned::Chunks(chunks) => mask.bitand(&chunks.intersect_by_rank(&result)),
                };
                if let Some(scheduler) = self.plan.scheduler() {
                    scheduler
                        .report_selectivity(index, self.mask.true_count() as f64 / input as f64);
                }
            }
            Some(Current::Projection) | None => {}
        }
        Ok(())
    }

    /// Narrows the mask to the rows whose zones `pruning`'s proof keeps.
    fn prune(
        &mut self,
        pruning: &ZonedPlan,
        zone_map: &ZoneMap,
        core: &mut Core,
    ) -> VortexResult<()> {
        let proof = pruning
            .proof()
            .ok_or_else(|| vortex_err!("Zone pruning plan has no proof"))?;
        // The zones a proof prunes are proven once per scan, unless they may change as it runs.
        let key = Arc::as_ptr(proof).cast::<()>() as usize;
        let pruned = match core.pruned.get(&key) {
            Some(pruned) if !proof.is_dynamic() => pruned.clone(),
            _ => {
                let pruned = proof.pruned(zone_map, &core.session)?;
                if !proof.is_dynamic() {
                    core.pruned.insert(key, pruned.clone());
                }
                pruned
            }
        };
        let zone_len = pruning.zone_len();
        let rows = &self.rows;
        let len = usize::try_from(rows.end - rows.start)?;
        let zones = pruned.slice(
            usize::try_from(rows.start / zone_len)?..usize::try_from(rows.end.div_ceil(zone_len))?,
        );
        if zones.all_false() {
            return Ok(());
        }
        let kept = if zones.all_true() {
            Mask::new_false(len)
        } else {
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
            Mask::from_buffer(bits.freeze())
        };
        let mask = std::mem::replace(&mut self.mask, Mask::new_false(0));
        self.mask = mask.bitand(&kept);
        Ok(())
    }

    /// Compiles the next stage that has rows to read, and returns its output port, or `None`
    /// once the split is done.
    /// Reads ahead every segment the conjuncts and the projection read for the rows the zones
    /// kept, so the reads overlap the stages that run before the ones that read them.
    fn prefetch(&self, core: &mut Core, split: usize) -> VortexResult<()> {
        let rows = &self.rows;
        let mask = &self.mask;
        let mut segments = Vec::new();
        let mut visit = |key: Shared, read: Range<u64>| {
            let Shared::Segment(segment) = key else {
                return;
            };
            let start = read.start.max(rows.start) - rows.start;
            let end = read.end.min(rows.end).saturating_sub(rows.start);
            if start < end
                && let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end))
                && !mask.slice(start..end).all_false()
            {
                segments.push(segment);
            }
        };
        for index in 0..self.plan.conjunct_count() {
            self.plan
                .conjunct(index)?
                .reach(rows.clone(), &Reach::Offset(0), &mut visit)?;
        }
        self.plan
            .projection()?
            .reach(rows.clone(), &Reach::Offset(0), &mut visit)?;
        core.prefetch(split, segments);
        Ok(())
    }

    pub(crate) fn next_stage(
        &mut self,
        core: &mut Core,
        slot: (usize, usize),
    ) -> VortexResult<Option<PortId>> {
        loop {
            if self.mask.all_false() {
                if self.phase == Phase::Conjuncts && !self.reported {
                    self.reported = true;
                    core.split_projects(slot.1, false);
                }
                self.phase = Phase::Done;
                return Ok(None);
            }
            match self.phase {
                Phase::Pruning(index) if index == self.plan.conjunct_count() => {
                    self.phase = Phase::Conjuncts;
                    self.prefetch(core, slot.1)?;
                }
                Phase::Pruning(index) => {
                    self.phase = Phase::Pruning(index + 1);
                    let Some(pruning) = self.plan.pruning(index, &core.session)? else {
                        continue;
                    };
                    let pruning = pruning.as_::<Zoned>().clone();
                    let zones = pruning.zones_plan()?;
                    let count = zones.row_count();
                    if let Some(zone_map) = core.zone_maps.get(&zones.as_ptr_key()).cloned() {
                        self.prune(&pruning, &zone_map, core)?;
                        continue;
                    }
                    let all = Mask::new_true(usize::try_from(count)?);
                    self.current = Some(Current::Zones(pruning));
                    if let Some(port) = core.compile_stage(&zones, 0..count, &all, slot)? {
                        return Ok(Some(port));
                    }
                    self.finish_stage(core)?;
                }
                Phase::Conjuncts => {
                    let next = self
                        .plan
                        .scheduler()
                        .and_then(|scheduler| scheduler.next_conjunct(&self.remaining));
                    let Some(index) = next else {
                        self.phase = Phase::Projecting;
                        continue;
                    };
                    self.remaining.set(index, false);
                    let conjunct = self.plan.conjunct(index)?;
                    let (spawned, mask) = if self.mask.density() < self.plan.dense_threshold() {
                        (Spawned::Selected, self.mask.clone())
                    } else {
                        let chunks = chunks_with_selected_rows(&conjunct, &self.rows, &self.mask)?;
                        (Spawned::Chunks(chunks.clone()), chunks)
                    };
                    self.current = Some(Current::Conjunct(index, spawned));
                    if let Some(port) =
                        core.compile_stage(&conjunct, self.rows.clone(), &mask, slot)?
                    {
                        return Ok(Some(port));
                    }
                    self.finish_stage(core)?;
                }
                Phase::Projecting => {
                    if !self.reported {
                        self.reported = true;
                        core.split_projects(slot.1, true);
                    }
                    self.phase = Phase::Done;
                    self.current = Some(Current::Projection);
                    let projection = self.plan.projection()?;
                    return core.compile_stage(&projection, self.rows.clone(), &self.mask, slot);
                }
                Phase::Done => return Ok(None),
            }
        }
    }
}
/// The mask over `rows` selecting every row of the chunks of `plan` in which `mask` selects a
/// row, and no row of the others.
///
/// A conjunct spawned under it runs over whole chunks, as a dense result is cheaper to
/// intersect with than a dense mask is to filter by, while the chunks nothing selects, such as
/// those the zones pruned, are skipped.
fn chunks_with_selected_rows(plan: &PlanRef, rows: &Range<u64>, mask: &Mask) -> VortexResult<Mask> {
    let len = mask.len();
    let mut starts = Vec::new();
    chunk_starts(plan, rows, 0, &mut starts)?;
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|&start| start > 0 && start < len);
    if starts.is_empty() {
        return Ok(Mask::new_true(len));
    }
    let mut bits = BitBufferMut::with_capacity(len);
    let mut start = 0;
    for end in starts.into_iter().chain([len]) {
        bits.append_n(!mask.slice(start..end).all_false(), end - start);
        start = end;
    }
    Ok(Mask::from_buffer(bits.freeze()))
}

/// Collects where the chunks of `plan` overlapping `rows` begin, as positions from the start
/// of the outermost range, descending through the operators that keep their child's row domain.
/// `offset` is where `rows` begins in that frame.
fn chunk_starts(
    plan: &PlanRef,
    rows: &Range<u64>,
    offset: usize,
    starts: &mut Vec<usize>,
) -> VortexResult<()> {
    if let Some(concat) = plan.as_opt::<Concat>() {
        let offsets = concat.row_offsets();
        let first = offsets
            .partition_point(|&o| o <= rows.start)
            .saturating_sub(1);
        let end = offsets.partition_point(|&o| o < rows.end);
        for index in first..end {
            let chunk_start = offsets[index];
            let chunk_end = offsets
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| concat.row_count());
            let local = rows.start.max(chunk_start)..rows.end.min(chunk_end);
            if local.start >= local.end {
                continue;
            }
            let position = offset + usize::try_from(local.start - rows.start)?;
            starts.push(position);
            chunk_starts(
                &concat.child_required(index)?,
                &(local.start - chunk_start..local.end - chunk_start),
                position,
                starts,
            )?;
        }
        return Ok(());
    }
    let children: Vec<PlanRef> = if let Some(eval) = plan.as_opt::<Eval>() {
        vec![eval.child_plan()?]
    } else if let Some(filter) = plan.as_opt::<Filter>() {
        vec![filter.child_plan()?]
    } else if let Some(zoned) = plan.as_opt::<Zoned>() {
        zoned.data_plan()?.into_iter().collect()
    } else if let Some(take) = plan.as_opt::<Take>() {
        vec![take.codes()?]
    } else if plan.is::<Pack>() {
        plan.children().iter().collect::<VortexResult<_>>()?
    } else {
        return Ok(());
    };
    for child in &children {
        chunk_starts(child, rows, offset, starts)?;
    }
    Ok(())
}
