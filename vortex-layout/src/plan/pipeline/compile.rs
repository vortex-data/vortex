// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines, and counting the readers of each segment so segments read
//! more than once are decoded once.

use std::ops::Range;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Operator;
use super::Source;
use super::ops::PortSource;
use super::ops::ScanSource;
use super::ops::SelectStage;
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
        let split = self.core.split;
        Ok(Some(match self.claim(scan, split) {
            None => Chain::new(ScanSource::new(scan.clone(), slice, filter)),
            Some(port) => {
                let chain = Chain {
                    source: Box::new(PortSource),
                    stages: Vec::new(),
                    inlets: vec![port],
                };
                if slice.is_none() && filter.is_none() {
                    chain
                } else {
                    chain.with(SelectStage::new(slice, filter))
                }
            }
        }))
    }

    /// Claims split `split`'s reader of `scan`'s segment: its port when the segment is
    /// shared, building the pipeline that decodes it for the first reader and a port for every
    /// reader in every split not yet finished, or `None` when this is its only reader.
    fn claim(&mut self, scan: &SegmentScanPlan, split: usize) -> Option<PortId> {
        if split == usize::MAX {
            return None;
        }
        let segment = scan.segment_id();
        let shares = &mut self.core.shares;
        let ports = &mut shares.ports[split];
        if let Some(waiting) = ports.get_mut(&segment) {
            let port = waiting.pop();
            if waiting.is_empty() {
                ports.remove(&segment);
            }
            return port;
        }
        let ranges = std::mem::take(shares.reaches.get_mut(*segment as usize)?);
        let mut readers: SmallVec<[usize; 4]> = SmallVec::new();
        for rows in &ranges {
            readers.extend(
                shares
                    .overlapping(rows)
                    .filter(|&reader| !shares.finished[reader] && shares.overlaps(reader, rows)),
            );
        }
        if readers.len() <= 1 {
            return None;
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
                    .entry(segment)
                    .or_default()
                    .push(port);
            }
        }
        self.core.add_pipeline(
            Chain::new(ScanSource::new(scan.clone(), None, None)),
            outlets,
        );
        mine
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
    /// By split: the unclaimed ports of each shared segment the split reads.
    ports: Vec<FxHashMap<SegmentId, SmallVec<[PortId; 1]>>>,
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
        let reaches = &mut self.reaches;
        plan.reach(rows, &Reach::Offset(0), &mut |segment, rows| {
            record(reaches, segment, rows)
        })
    }

    /// Keeps only the segments more than one reader reads. The rest are read by the one
    /// pipeline that needs them.
    pub(crate) fn retain_shared(&mut self) {
        let mut reaches = std::mem::take(&mut self.reaches);
        for ranges in &mut reaches {
            let shared = match ranges.as_slice() {
                [] => false,
                // One reader per split overlapping the range: shared when the range crosses a
                // split boundary.
                [rows] => self
                    .overlapping(rows)
                    .filter(|&split| self.overlaps(split, rows))
                    .nth(1)
                    .is_some(),
                _ => true,
            };
            if !shared {
                *ranges = SmallVec::new();
            }
        }
        self.reaches = reaches;
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

/// Adds a reader of `segment` reading it for `rows`.
fn record(reaches: &mut Vec<SmallVec<[Range<u64>; 1]>>, segment: SegmentId, rows: Range<u64>) {
    let index = *segment as usize;
    if index >= reaches.len() {
        reaches.resize_with(index + 1, SmallVec::new);
    }
    reaches[index].push(rows);
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
