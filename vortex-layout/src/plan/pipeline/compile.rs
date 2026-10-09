// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines, and counting the readers of each segment so segments read
//! more than once are decoded once.

use std::ops::Range;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Operator;
use super::Source;
use super::ops::OnceSource;
use super::ops::PortSource;
use super::ops::ScanSource;
use super::ops::SelectStage;
use super::ops::WholeStage;
use super::port::PortId;
use super::port::Reader;
use super::scan::Core;
use crate::plan::ConcatPlan;
use crate::plan::PlanRef;
use crate::plan::SegmentScanPlan;
use crate::segments::SegmentId;

/// A pipeline under construction: a source and the stages after it, not yet given outlets, so a
/// parent plan can add stages before it becomes a pipeline.
pub struct Chain {
    pub(crate) source: Box<dyn Source>,
    pub(crate) stages: Vec<Box<dyn Operator>>,
    pub(crate) inlets: Vec<PortId>,
}

impl Chain {
    /// A chain of `source` alone.
    pub fn new(source: impl Source + 'static) -> Self {
        Self {
            source: Box::new(source),
            stages: Vec::new(),
            inlets: Vec::new(),
        }
    }

    /// A chain passing the batches of `port` through.
    pub(crate) fn port(port: PortId) -> Self {
        Self {
            source: Box::new(PortSource),
            stages: Vec::new(),
            inlets: vec![port],
        }
    }

    /// This chain with `stage` after its last stage.
    pub fn with(mut self, stage: impl Operator + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }
}

impl Core {
    /// Compiles `plan` over `rows`, restricted to `mask`. See [`Compiler::compile`].
    pub(crate) fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        Compiler { core: self }.compile(plan, rows, mask)
    }
}

/// What a plan compiles its children through. See
/// [`PlanVTable::compile`](crate::plan::PlanVTable::compile).
pub struct Compiler<'a> {
    pub(crate) core: &'a mut Core,
}

impl Compiler<'_> {
    /// Compiles `plan` over `rows` of its domain, restricted to `mask`, into a chain producing
    /// the rows `mask` selects, in order, or `None` when that is no row.
    pub fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        if rows.start >= rows.end {
            return Ok(None);
        }
        plan.compile(rows, mask, self)
    }

    /// A chain reading `chains` through `source`: each chain becomes a pipeline writing a new
    /// port, inlet `i` of `source` reading chain `i`, with the capacity `source` asks for.
    pub fn join(&mut self, chains: Vec<Chain>, source: impl Source + 'static) -> Chain {
        let inlets = chains
            .into_iter()
            .enumerate()
            .map(|(index, chain)| {
                let port = self
                    .core
                    .arena
                    .create(source.capacity(index), None, Reader::Unclaimed);
                self.core.add_pipeline(chain, smallvec::smallvec![port]);
                port
            })
            .collect();
        Chain {
            source: Box::new(source),
            stages: Vec::new(),
            inlets,
        }
    }

    /// The global row index of the scanned plan's row zero.
    pub fn row_offset(&self) -> u64 {
        self.core.row_offset
    }

    /// The session used for decoding and evaluation.
    pub fn session(&self) -> &VortexSession {
        &self.core.session
    }

    /// The rows `rows` of `scan`'s segment, filtered by `filter`, read by the split compiling.
    /// A segment several readers read is decoded once and shared.
    pub(crate) fn scan(
        &mut self,
        scan: &SegmentScanPlan,
        rows: Range<u64>,
        filter: Option<Mask>,
    ) -> VortexResult<Option<Chain>> {
        let filter = match filter {
            Some(filter) if filter.all_false() => return Ok(None),
            Some(filter) if filter.all_true() => None,
            filter => filter,
        };
        let slice = if rows.start == 0 && rows.end == scan.row_count() {
            None
        } else {
            Some(usize::try_from(rows.start)?..usize::try_from(rows.end)?)
        };
        let key = Shared::Segment(scan.segment_id());
        let decode = |_: &mut Self| Ok(Chain::new(ScanSource::new(scan.clone(), None, None)));
        Ok(Some(match self.claim(key, decode)? {
            None => Chain::new(ScanSource::new(scan.clone(), slice, filter)),
            Some(port) => {
                let chain = Chain::port(port);
                if slice.is_none() && filter.is_none() {
                    chain
                } else {
                    chain.with(SelectStage::new(slice, filter))
                }
            }
        }))
    }

    /// Compiles `plan` over its whole domain into a chain producing it as one shared array.
    ///
    /// With `share`, the scan reads it once however many readers in however many splits need
    /// it, as a dictionary's values are needed by every split reading its codes: the first
    /// reader builds the pipeline, and each reader, this one included, reads its own port. The
    /// array is dropped once the last split that may read it has finished.
    pub fn whole(&mut self, plan: &PlanRef, share: bool) -> VortexResult<Chain> {
        if share && let Some(port) = self.claim(Shared::plan(plan), |c| c.whole_chain(plan))? {
            return Ok(Chain::port(port));
        }
        self.whole_chain(plan)
    }

    fn whole_chain(&mut self, plan: &PlanRef) -> VortexResult<Chain> {
        let len = plan.row_count();
        let chain = match self.compile(plan, 0..len, &Mask::new_true(usize::try_from(len)?))? {
            Some(chain) => chain,
            None => Chain::new(OnceSource::new(Canonical::empty(plan.dtype()).into_array())),
        };
        Ok(chain.with(WholeStage::new(plan.dtype().clone())))
    }

    /// Claims the split compiling's reader of `key`: its port when `key` is shared, building
    /// the pipeline `build` makes for the first reader with a port for every reader in every
    /// split not yet finished, or `None` when this is its only reader.
    fn claim(
        &mut self,
        key: Shared,
        build: impl FnOnce(&mut Self) -> VortexResult<Chain>,
    ) -> VortexResult<Option<PortId>> {
        let split = self.core.split;
        if split == usize::MAX {
            return Ok(None);
        }
        let shares = &mut self.core.shares;
        let ports = &mut shares.ports[split];
        if let Some(waiting) = ports.get_mut(&key) {
            let port = waiting.pop();
            if waiting.is_empty() {
                ports.remove(&key);
            }
            return Ok(port);
        }
        let Some(ranges) = shares.take_reaches(key) else {
            return Ok(None);
        };
        let mut readers: SmallVec<[usize; 4]> = SmallVec::new();
        for rows in &ranges {
            readers.extend(
                shares
                    .overlapping(rows)
                    .filter(|&reader| !shares.finished[reader] && shares.overlaps(reader, rows)),
            );
        }
        if readers.len() <= 1 {
            return Ok(None);
        }
        let mut outlets: SmallVec<[PortId; 1]> = SmallVec::with_capacity(readers.len());
        let mut mine = None;
        for reader in readers {
            let port = self.core.arena.create(usize::MAX, None, Reader::Unclaimed);
            outlets.push(port);
            if reader == split && mine.is_none() {
                mine = Some(port);
            } else {
                self.core.shares.ports[reader]
                    .entry(key)
                    .or_default()
                    .push(port);
            }
        }
        // What the shared pipeline reads is read by it alone, for no split.
        self.core.split = usize::MAX;
        let chain = build(self);
        self.core.split = split;
        self.core.add_pipeline(chain?, outlets);
        Ok(mine)
    }
}

/// What several readers can share: a decoded segment, or a plan's output over its whole domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shared {
    /// The whole decoded segment.
    Segment(SegmentId),
    /// The output of the plan at this address, read whole. The scan holds the plan, so the
    /// address names it for as long as the scan runs.
    Plan(usize),
}

impl Shared {
    /// The key of `plan`'s whole output.
    pub fn plan(plan: &PlanRef) -> Self {
        Shared::Plan(plan.as_ptr_key())
    }
}

/// Where each segment read more than once is read from, and the ports of the segments whose
/// decoding pipeline is built.
#[derive(Default)]
pub(crate) struct Shares {
    /// The splits' row ranges, in split order.
    splits: Vec<Range<u64>>,
    /// Whether the splits are sorted and disjoint, so the splits overlapping a range are found
    /// by binary search.
    ordered: bool,
    /// Splits that have finished, and so will claim nothing.
    finished: Vec<bool>,
    /// By segment id, which a file numbers densely: for each segment more than one reader will
    /// read and whose decoding pipeline is not built yet, the range of plan rows each of its
    /// readers reads it for. Every split overlapping a range reads it once for that range.
    reaches: Vec<SmallVec<[Range<u64>; 1]>>,
    /// The same for plans read whole, by address.
    plan_reaches: FxHashMap<usize, SmallVec<[Range<u64>; 1]>>,
    /// By split: the unclaimed ports of each shared output the split reads.
    ports: Vec<FxHashMap<Shared, SmallVec<[PortId; 1]>>>,
}

impl Shares {
    pub(crate) fn new(splits: Vec<Range<u64>>) -> Self {
        let ordered = splits.windows(2).all(|pair| pair[0].end <= pair[1].start);
        Self {
            finished: vec![false; splits.len()],
            ports: (0..splits.len()).map(|_| FxHashMap::default()).collect(),
            splits,
            ordered,
            reaches: Vec::new(),
            plan_reaches: FxHashMap::default(),
        }
    }

    /// The splits overlapping `rows`.
    fn overlapping(&self, rows: &Range<u64>) -> Range<usize> {
        if self.ordered {
            let first = self.splits.partition_point(|split| split.end <= rows.start);
            let end = self.splits.partition_point(|split| split.start < rows.end);
            return first..end.max(first);
        }
        let mut overlapping = self
            .splits
            .iter()
            .enumerate()
            .filter(|(_, split)| split.start < rows.end && rows.start < split.end)
            .map(|(index, _)| index);
        match overlapping.next() {
            Some(first) => first..overlapping.next_back().unwrap_or(first) + 1,
            None => 0..0,
        }
    }

    fn overlaps(&self, split: usize, rows: &Range<u64>) -> bool {
        let split = &self.splits[split];
        split.start < rows.end && rows.start < split.end
    }

    /// Records the ranges `plan` reads each of its segments for, read over `rows` by every
    /// split overlapping them. Call once per plan a split stage runs, before the scan starts,
    /// then [`retain_shared`](Self::retain_shared).
    pub(crate) fn add(&mut self, plan: &PlanRef, rows: Range<u64>) -> VortexResult<()> {
        plan.reach(rows, &Reach::Offset(0), &mut |key, rows| {
            self.record(key, rows)
        })
    }

    /// Records that every split reads the segments of `plan` once, over its whole domain.
    pub(crate) fn add_every_split(&mut self, plan: &PlanRef, rows: Range<u64>) -> VortexResult<()> {
        let all = 0..u64::MAX;
        plan.reach(rows, &Reach::Fixed(all), &mut |key, rows| {
            self.record(key, rows)
        })
    }

    /// Keeps only the segments more than one reader reads. The rest are read by the one
    /// pipeline that needs them.
    pub(crate) fn retain_shared(&mut self) {
        let mut reaches = std::mem::take(&mut self.reaches);
        for ranges in &mut reaches {
            if !self.is_shared(ranges) {
                *ranges = SmallVec::new();
            }
        }
        self.reaches = reaches;
        let plans = std::mem::take(&mut self.plan_reaches);
        self.plan_reaches = plans
            .into_iter()
            .filter(|(_, ranges)| self.is_shared(ranges))
            .collect();
    }

    /// Whether readers reading for `ranges` are in more than one split, or more than one in a
    /// split.
    fn is_shared(&self, ranges: &[Range<u64>]) -> bool {
        match ranges {
            [] => false,
            // One reader per split overlapping the range: shared when the range crosses a split
            // boundary.
            [rows] => self
                .overlapping(rows)
                .filter(|&split| self.overlaps(split, rows))
                .nth(1)
                .is_some(),
            _ => true,
        }
    }

    /// Adds a reader of `key` reading it for `rows`.
    fn record(&mut self, key: Shared, rows: Range<u64>) {
        match key {
            Shared::Segment(segment) => {
                let index = *segment as usize;
                if index >= self.reaches.len() {
                    self.reaches.resize_with(index + 1, SmallVec::new);
                }
                self.reaches[index].push(rows);
            }
            Shared::Plan(plan) => self.plan_reaches.entry(plan).or_default().push(rows),
        }
    }

    /// The ranges `key` is read for, if it is shared and its pipeline is not built yet.
    fn take_reaches(&mut self, key: Shared) -> Option<SmallVec<[Range<u64>; 1]>> {
        let ranges = match key {
            Shared::Segment(segment) => std::mem::take(self.reaches.get_mut(*segment as usize)?),
            Shared::Plan(plan) => self.plan_reaches.remove(&plan)?,
        };
        (!ranges.is_empty()).then_some(ranges)
    }

    /// Drops the ports split `split` did not claim: its readers that never came, as a chunk
    /// its mask ruled out or a stage it never ran.
    pub(crate) fn finish(&mut self, split: usize, arena: &mut super::port::Arena) {
        self.finished[split] = true;
        for (_, ports) in std::mem::take(&mut self.ports[split]) {
            for port in ports {
                arena.drop_reader(port);
            }
        }
    }
}

/// How a plan's rows map to the rows of the plan a scan's splits range over.
#[derive(Clone, Debug)]
pub enum Reach {
    /// Row `r` is scanned row `r + offset`.
    Offset(u64),
    /// Every row is read for these scanned rows, as a dictionary's values are for its codes.
    Fixed(Range<u64>),
}

impl Reach {
    /// The scanned rows `rows` are read for.
    pub fn root(&self, rows: &Range<u64>) -> Range<u64> {
        match self {
            Reach::Offset(offset) => rows.start + offset..rows.end + offset,
            Reach::Fixed(root) => root.clone(),
        }
    }

    /// The mapping of a child whose row zero is this plan's row `by`.
    pub fn shift(&self, by: u64) -> Self {
        match self {
            Reach::Offset(offset) => Reach::Offset(offset + by),
            Reach::Fixed(root) => Reach::Fixed(root.clone()),
        }
    }

    /// The mapping of a child read whole for `rows` of this plan, as dictionary values are.
    pub fn fixed(&self, rows: &Range<u64>) -> Self {
        Reach::Fixed(self.root(rows))
    }
}

/// The chunks of `concat` overlapping `rows`: each chunk's index, where it starts, and the rows
/// of `rows` it holds, in the concatenation's domain. Found by binary search, so a split pays
/// for the chunks it overlaps, not for the chunks of the file.
pub(crate) fn overlapping<'a>(
    concat: &'a ConcatPlan,
    rows: &Range<u64>,
) -> impl Iterator<Item = (usize, u64, Range<u64>)> + 'a {
    let offsets = concat.row_offsets();
    let first = offsets
        .partition_point(|&offset| offset <= rows.start)
        .saturating_sub(1);
    let end = offsets.partition_point(|&offset| offset < rows.end);
    let rows = rows.clone();
    (first..end).filter_map(move |index| {
        let start = offsets[index];
        let chunk_end = offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| concat.row_count());
        let local = rows.start.max(start)..rows.end.min(chunk_end);
        (local.start < local.end).then_some((index, start, local))
    })
}

/// Unclaimed shared ports, for tests.
#[cfg(test)]
impl Shares {
    pub(crate) fn pending(&self) -> usize {
        self.ports
            .iter()
            .flat_map(|ports| ports.values())
            .map(SmallVec::len)
            .sum()
    }
}
