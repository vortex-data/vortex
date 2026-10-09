// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiles plans into pipelines, and counts the readers of each segment so segments read more
//! than once are decoded once.

use std::ops::Range;

use smallvec::SmallVec;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::Operator;
use super::Source;
use super::ops::ConcatSource;
use super::ops::EvalStage;
use super::ops::ListPackSource;
use super::ops::MaskStage;
use super::ops::OnceSource;
use super::ops::PackSource;
use super::ops::PortSource;
use super::ops::ScanSource;
use super::ops::SelectStage;
use super::ops::TakeSource;
use super::ops::WrapStage;
use super::ops::empty_struct;
use super::port::PortId;
use super::port::Reader;
use super::scan::Core;
use crate::plan::Concat;
use crate::plan::ConcatPlan;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::ListPack;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::Query;
use crate::plan::RowIdx;
use crate::plan::SegmentScan;
use crate::plan::SegmentScanPlan;
use crate::plan::Take;
use crate::plan::Zoned;
use crate::segments::SegmentId;

/// A pipeline under construction: a source and the stages after it, not yet given outlets, so a
/// parent can add stages before it is made a pipeline.
pub(crate) struct Chain {
    pub(crate) source: Box<dyn Source>,
    pub(crate) stages: Vec<Box<dyn Operator>>,
    pub(crate) inlets: Vec<PortId>,
}

impl Chain {
    fn new(source: impl Source + 'static) -> Self {
        Self {
            source: Box::new(source),
            stages: Vec::new(),
            inlets: Vec::new(),
        }
    }

    fn with(mut self, stage: impl Operator + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
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
    /// By split: the unclaimed ports of the shared segments the split reads. A split reads few
    /// shared segments, so they are found by a short linear search.
    ports: Vec<SmallVec<[(SegmentId, PortId); 2]>>,
}

impl Shares {
    pub(crate) fn new(splits: Vec<Range<u64>>) -> Self {
        let ordered = splits.windows(2).all(|pair| pair[0].end <= pair[1].start);
        Self {
            finished: vec![false; splits.len()],
            ports: vec![SmallVec::new(); splits.len()],
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
        reach(plan, rows, Reach::Offset(0), &mut |segment, rows| {
            record(reaches, segment, rows)
        })
    }

    /// Records that every split reads the segments of `plan` once, over its whole domain.
    pub(crate) fn add_every_split(&mut self, plan: &PlanRef, rows: Range<u64>) -> VortexResult<()> {
        let all = 0..u64::MAX;
        let reaches = &mut self.reaches;
        reach(plan, rows, Reach::Fixed(all), &mut |segment, rows| {
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
        for (_, port) in std::mem::take(&mut self.ports[split]) {
            arena.drop_reader(port);
        }
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

/// How a plan's rows map to the rows of the plan a split ranges over.
#[derive(Clone)]
enum Reach {
    /// Row `r` is root row `r + offset`.
    Offset(u64),
    /// Every row is read for these root rows, as a dictionary's values are for its codes.
    Fixed(Range<u64>),
}

impl Reach {
    fn root(&self, rows: &Range<u64>) -> Range<u64> {
        match self {
            Reach::Offset(offset) => rows.start + offset..rows.end + offset,
            Reach::Fixed(root) => root.clone(),
        }
    }

    fn shift(&self, by: u64) -> Self {
        match self {
            Reach::Offset(offset) => Reach::Offset(offset + by),
            Reach::Fixed(root) => Reach::Fixed(root.clone()),
        }
    }
}

/// Visits every segment compiling `plan` over `rows` could read, with the root rows it is
/// read for. Visits each plan node once, so counting a scan's readers costs one pass over its
/// plans, not one per split.
///
/// Lists are not visited: the range of a list's elements is known only once its offsets are
/// read, so a list's segments are read by the list alone.
fn reach(
    plan: &PlanRef,
    rows: Range<u64>,
    at: Reach,
    visit: &mut dyn FnMut(SegmentId, Range<u64>),
) -> VortexResult<()> {
    if rows.start >= rows.end {
        return Ok(());
    }
    if let Some(scan) = plan.as_opt::<SegmentScan>() {
        visit(scan.segment_id(), at.root(&rows));
    } else if let Some(filter) = plan.as_opt::<Filter>() {
        reach(&filter.child_plan()?, rows, at, visit)?;
    } else if let Some(eval) = plan.as_opt::<Eval>() {
        reach(&eval.child_plan()?, rows, at, visit)?;
    } else if let Some(concat) = plan.as_opt::<Concat>() {
        for (index, start, local) in overlapping(concat, &rows) {
            reach(
                &concat.child_required(index)?,
                local.start - start..local.end - start,
                at.shift(start),
                visit,
            )?;
        }
    } else if plan.is::<Pack>() {
        for child in plan.children().iter() {
            reach(&child?, rows.clone(), at.clone(), visit)?;
        }
    } else if let Some(take) = plan.as_opt::<Take>() {
        let values = take.values()?;
        let len = values.row_count();
        reach(&values, 0..len, Reach::Fixed(at.root(&rows)), visit)?;
        reach(&take.codes()?, rows, at, visit)?;
    } else if let Some(zoned) = plan.as_opt::<Zoned>() {
        if let Some(data) = zoned.data_plan()? {
            reach(&data, rows, at, visit)?;
        }
    } else if plan.is::<ListPack>() || plan.is::<RowIdx>() {
    } else if plan.is::<Query>() {
        vortex_bail!("A Query plan runs only at the root of a scan");
    } else {
        vortex_bail!("Plan {} has no pipeline implementation", plan.id());
    }
    Ok(())
}

/// The chunks of `concat` overlapping `rows`: each chunk's index, where it starts, and the rows
/// of `rows` it holds, in the concatenation's domain. Found by binary search, so a split pays
/// for the chunks it overlaps, not for the chunks of the file.
fn overlapping<'a>(
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

fn len_of(rows: &Range<u64>) -> VortexResult<usize> {
    Ok(usize::try_from(rows.end - rows.start)?)
}

/// The slice of a whole segment holding `rows`, or `None` for all of it.
fn slice_of(plan: &SegmentScanPlan, rows: &Range<u64>) -> VortexResult<Option<Range<usize>>> {
    if rows.start == 0 && rows.end == plan.row_count() {
        return Ok(None);
    }
    Ok(Some(
        usize::try_from(rows.start)?..usize::try_from(rows.end)?,
    ))
}

impl Core {
    /// Claims split `split`'s reader of `scan`'s segment: its port when the segment is
    /// shared, building the pipeline that decodes it for the first reader and a port for every
    /// reader in every split not yet finished, or `None` when this is its only reader.
    fn claim(&mut self, scan: &SegmentScanPlan, split: usize) -> Option<PortId> {
        if split == usize::MAX {
            return None;
        }
        let segment = scan.segment_id();
        let shares = &mut self.shares;
        let ports = &mut shares.ports[split];
        if let Some(position) = ports.iter().position(|(shared, _)| *shared == segment) {
            return Some(ports.swap_remove(position).1);
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
            let port = self.arena.create(usize::MAX, None, Reader::Unclaimed);
            outlets.push(port);
            if reader == split && mine.is_none() {
                mine = Some(port);
            } else {
                self.shares.ports[reader].push((segment, port));
            }
        }
        self.add_pipeline(
            Chain::new(ScanSource::new(scan.clone(), None, None)),
            outlets,
        );
        mine
    }

    /// The rows `rows` of a segment, filtered by `filter`.
    fn scan(
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
        let slice = slice_of(scan, &rows)?;
        Ok(Some(match self.claim(scan, self.split) {
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

    /// Makes each chain a pipeline writing a new port read by `reader`, with the capacity the
    /// reader asks for, and returns the ports in order.
    fn feed(&mut self, chains: Vec<Chain>, reader: &dyn Source) -> Vec<PortId> {
        chains
            .into_iter()
            .enumerate()
            .map(|(index, chain)| {
                let port = self
                    .arena
                    .create(reader.capacity(index), None, Reader::Unclaimed);
                self.add_pipeline(chain, smallvec::smallvec![port]);
                port
            })
            .collect()
    }

    /// A chain reading `chains` through `source`.
    fn join(&mut self, chains: Vec<Chain>, source: impl Source + 'static) -> Chain {
        let inlets = self.feed(chains, &source);
        Chain {
            source: Box::new(source),
            stages: Vec::new(),
            inlets,
        }
    }

    /// Compiles `plan` over `rows` of its domain into a chain producing the rows `mask`
    /// selects, in order, or `None` when that is no row.
    ///
    /// A bare segment scan produces every row of `rows`, as its plan says; a filter over it
    /// keeps the selected ones. Shared segments are claimed for the split compiling.
    pub(crate) fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        if rows.start >= rows.end {
            return Ok(None);
        }
        if let Some(scan) = plan.as_opt::<SegmentScan>() {
            return self.scan(scan, rows, None);
        }
        if let Some(filter) = plan.as_opt::<Filter>() {
            let child = filter.child_plan()?;
            if let Some(scan) = child.as_opt::<SegmentScan>() {
                return self.scan(scan, rows, Some(mask.clone()));
            }
            if mask.all_false() {
                return Ok(None);
            }
            let len = len_of(&rows)?;
            let predicate = plan.dtype().is_boolean();
            let chain = self.compile(&child, rows, &Mask::new_true(len))?;
            return Ok(chain.map(|chain| {
                if mask.all_true() {
                    chain
                } else {
                    chain.with(MaskStage::new(mask.clone(), predicate))
                }
            }));
        }
        if let Some(eval) = plan.as_opt::<Eval>() {
            let chain = self.compile(&eval.child_plan()?, rows, mask)?;
            return Ok(chain.map(|chain| chain.with(EvalStage::new(eval.expression().clone()))));
        }
        if let Some(concat) = plan.as_opt::<Concat>() {
            return self.concat(concat, rows, mask);
        }
        if let Some(pack) = plan.as_opt::<Pack>() {
            let count = pack.children().len();
            if count == 0 {
                let array = empty_struct(
                    pack.fields().clone(),
                    pack.dtype().nullability(),
                    mask.true_count(),
                )?;
                return Ok(Some(Chain::new(OnceSource::new(array))));
            }
            if mask.all_false() {
                return Ok(None);
            }
            let mut chains = Vec::with_capacity(count);
            for child in pack.children().iter() {
                chains.push(
                    self.compile(&child?, rows.clone(), mask)?
                        .ok_or_else(|| vortex_err!("A Pack field produced no rows"))?,
                );
            }
            let nullable = pack.dtype().is_nullable();
            if count == 1 && !nullable {
                let chain = chains.remove(0);
                return Ok(Some(
                    chain.with(WrapStage::new(pack.fields().names().clone())),
                ));
            }
            let source = PackSource::new(pack.fields().clone(), nullable, count);
            return Ok(Some(self.join(chains, source)));
        }
        if plan.is::<RowIdx>() {
            let offset = self.row_offset;
            let indices = Buffer::from_iter(rows.start + offset..rows.end + offset).into_array();
            let indices = if mask.all_true() {
                indices
            } else if mask.all_false() {
                return Ok(None);
            } else {
                indices.filter(mask.clone())?
            };
            return Ok(Some(Chain::new(OnceSource::new(indices))));
        }
        if let Some(take) = plan.as_opt::<Take>() {
            let values = take.values()?;
            let values_rows = 0..values.row_count();
            let Some(codes) = self.compile(&take.codes()?, rows, mask)? else {
                return Ok(None);
            };
            if let Some(cached) = take.cached_values() {
                let source = TakeSource::new(take.clone(), Some(cached));
                return Ok(Some(self.join(vec![codes], source)));
            }
            let len = len_of(&values_rows)?;
            let values = match self.compile(&values, values_rows, &Mask::new_true(len))? {
                Some(values) => values,
                None => Chain::new(OnceSource::new(
                    vortex_array::Canonical::empty(values.dtype()).into_array(),
                )),
            };
            let source = TakeSource::new(take.clone(), None);
            return Ok(Some(self.join(vec![codes, values], source)));
        }
        if let Some(list) = plan.as_opt::<ListPack>() {
            return self.list(list, rows, mask);
        }
        if let Some(zoned) = plan.as_opt::<Zoned>() {
            let Some(data) = zoned.data_plan()? else {
                vortex_bail!("A zone pruning plan runs only inside a Query");
            };
            return self.compile(&data, rows, mask);
        }
        if plan.is::<Query>() {
            vortex_bail!("A Query plan runs only at the root of a scan");
        }
        vortex_bail!("Plan {} has no pipeline implementation", plan.id())
    }

    fn concat(
        &mut self,
        concat: &ConcatPlan,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        let mut chains = Vec::new();
        for (index, start, local) in overlapping(concat, &rows) {
            let chunk = concat.child_required(index)?;
            let local_mask = mask.slice(
                usize::try_from(local.start - rows.start)?
                    ..usize::try_from(local.end - rows.start)?,
            );
            let chunk_rows = local.start - start..local.end - start;
            if local_mask.all_false() {
                continue;
            }
            if let Some(chain) = self.compile(&chunk, chunk_rows, &local_mask)? {
                chains.push(chain);
            }
        }
        Ok(match chains.len() {
            0 => None,
            1 => chains.pop(),
            count => Some(self.join(chains, ConcatSource::new(count))),
        })
    }

    fn list(
        &mut self,
        list: &crate::plan::ListPackPlan,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        let offsets = list.offsets()?;
        let validity = list.validity()?;
        let (Some(first), Some(last)) = (mask.first(), mask.last()) else {
            return Ok(None);
        };
        let read = rows.start + u64::try_from(first)?..rows.start + u64::try_from(last)? + 1;
        let mask = mask.slice(first..last + 1);
        let len = mask.len();
        let mut chains = vec![
            self.compile(&offsets, read.start..read.end + 1, &Mask::new_true(len + 1))?
                .ok_or_else(|| vortex_err!("List offsets produced no rows"))?,
        ];
        if let Some(validity) = &validity {
            chains.push(
                self.compile(validity, read, &Mask::new_true(len))?
                    .ok_or_else(|| vortex_err!("List validity produced no rows"))?,
            );
        }
        let source = ListPackSource::new(list.clone(), mask, validity.is_some());
        Ok(Some(self.join(chains, source)))
    }
}

/// Unclaimed shared ports, for tests.
#[cfg(test)]
impl Shares {
    pub(crate) fn pending(&self) -> usize {
        self.ports.iter().map(SmallVec::len).sum()
    }
}
