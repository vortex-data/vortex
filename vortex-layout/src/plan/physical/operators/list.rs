// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from offsets and the element ranges holding them.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::UNBOUNDED;
use super::join_inlet;
use super::take_rows;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

/// Joins a list's offsets, and its validity when nullable, then emits each run of selected lists
/// once their elements have arrived.
///
/// The offsets cover every row from the first selected list to the last, one more than the rows;
/// the validity covers the selected lists only. The elements inlet holds the element ranges
/// `ranges`, in order, which hold every selected list and may hold others: their elements are
/// dropped here.
pub(crate) struct ListSource {
    elements_dtype: DType,
    offsets_dtype: DType,
    nullability: Nullability,
    /// The lists to produce, over the rows the offsets cover.
    mask: Mask,
    /// The element ranges the elements inlet holds, or `None` when there is no elements inlet.
    ranges: Option<Vec<Range<u64>>>,
    lists: Option<Lists>,
}

/// The selected lists, once their offsets and validity have been joined.
struct Lists {
    /// Each selected list's first element, as a row of the elements inlet.
    starts: Buffer<u64>,
    /// Each selected list's output offsets: where it starts among the selected elements, and the
    /// end of the last.
    offsets: Buffer<u64>,
    validity: Option<ArrayRef>,
    /// Lists emitted so far.
    next: usize,
    /// The row of the elements inlet its next batch starts at.
    cursor: u64,
}

impl Lists {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The row of the elements inlet after the last element of list `index`.
    fn end(&self, index: usize) -> u64 {
        self.starts[index] + (self.offsets[index + 1] - self.offsets[index])
    }
}

impl ListSource {
    pub(crate) fn new(
        elements_dtype: DType,
        offsets_dtype: DType,
        nullability: Nullability,
        mask: Mask,
        ranges: Option<Vec<Range<u64>>>,
    ) -> Self {
        Self {
            elements_dtype,
            offsets_dtype,
            nullability,
            mask,
            ranges,
            lists: None,
        }
    }

    /// The batches inlet `inlet` holds: the offsets and validity are joined whole, and the
    /// elements of a list are taken only once all of them have arrived.
    pub(crate) fn capacity(_inlet: usize) -> usize {
        UNBOUNDED
    }

    fn joined_inlets(&self) -> usize {
        1 + usize::from(self.nullability.is_nullable())
    }

    fn elements_inlet(&self) -> Option<usize> {
        self.ranges.as_ref().map(|_| self.joined_inlets())
    }

    /// Joins the offsets and validity, and places each selected list in the elements inlet.
    fn start(&self, cx: &mut Cx<'_>) -> VortexResult<Lists> {
        let raw = join_inlet(cx, OFFSETS, &self.offsets_dtype)?
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(cx.exec())?;
        let raw = raw.as_slice::<u64>();
        vortex_ensure!(
            raw.len() == self.mask.len() + 1,
            "List offsets cover {} rows but the mask {}",
            raw.len().saturating_sub(1),
            self.mask.len()
        );
        let validity = self
            .nullability
            .is_nullable()
            .then(|| join_inlet(cx, VALIDITY, &DType::Bool(Nullability::NonNullable)))
            .transpose()?;

        let ranges = self.ranges.as_deref().unwrap_or_default();
        // Where each range starts among the inlet's rows.
        let mut range_rows = Vec::with_capacity(ranges.len());
        let mut total = 0u64;
        for range in ranges {
            range_rows.push(total);
            total += range.end - range.start;
        }
        let count = self.mask.true_count();
        let mut starts = BufferMut::<u64>::with_capacity(count);
        let mut offsets = BufferMut::<u64>::with_capacity(count + 1);
        offsets.push(0);
        let mut range = 0usize;
        // The inlet row after the last list placed, where an empty list is placed.
        let mut placed = 0u64;
        let mut select = |start: usize, end: usize| -> VortexResult<()> {
            let mut out = *offsets.last().unwrap_or(&0);
            for window in raw[start..=end].windows(2) {
                let (first, last) = (window[0], window[1]);
                vortex_ensure!(first <= last, "List offsets decrease");
                let row = if first == last {
                    placed
                } else {
                    while range < ranges.len() && ranges[range].end <= first {
                        range += 1;
                    }
                    let held = ranges.get(range).ok_or_else(|| {
                        vortex_err!("List elements {first}..{last} lie outside the read ranges")
                    })?;
                    vortex_ensure!(
                        held.start <= first && last <= held.end,
                        "List elements {first}..{last} lie outside the read ranges"
                    );
                    range_rows[range] + (first - held.start)
                };
                placed = row + (last - first);
                starts.push(row);
                out += last - first;
                offsets.push(out);
            }
            Ok(())
        };
        match self.mask.slices() {
            AllOr::All => select(0, self.mask.len())?,
            AllOr::None => {}
            AllOr::Some(selected) => {
                for &(start, end) in selected {
                    select(start, end)?;
                }
            }
        }
        Ok(Lists {
            starts: starts.freeze(),
            offsets: offsets.freeze(),
            validity,
            next: 0,
            cursor: 0,
        })
    }
}

impl Source for ListSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.lists.is_none() {
            for index in 0..self.joined_inlets() {
                if !cx.inlet(index).closed() {
                    return Ok(SourceReady::Blocked(SourceBlocked::Inlet(index)));
                }
            }
            self.lists = Some(self.start(cx)?);
        }
        let elements = self.elements_inlet();
        let lists = self
            .lists
            .as_mut()
            .ok_or_else(|| vortex_err!("List offsets are joined"))?;
        let Some(index) = elements else {
            return Ok(SourceReady::Ready);
        };
        let mut inlet = cx.inlet(index);
        if lists.next == lists.len() {
            // Every list emitted: the elements after the last one are dropped as they arrive,
            // until the inlet closes.
            while let Some(batch) = inlet.take() {
                lists.cursor += batch.len() as u64;
            }
            return Ok(if inlet.closed() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Inlet(index))
            });
        }
        if inlet.closed() {
            return Ok(SourceReady::Ready);
        }
        // The lists are emitted together, once all their elements are here: a column emitting a
        // batch per element segment would cut the struct it is a field of at every one of its
        // segments' edges, and a wide struct into many slivers.
        let last = lists.len() - 1;
        let complete = lists.end(last) <= lists.cursor + inlet.rows() as u64;
        Ok(if complete {
            SourceReady::Ready
        } else {
            SourceReady::Blocked(SourceBlocked::Inlet(index))
        })
    }
}

impl Operator for ListSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let elements_inlet = self.elements_inlet();
        let lists = self
            .lists
            .as_mut()
            .ok_or_else(|| vortex_err!("List computed before its offsets were joined"))?;
        if lists.next == lists.len() {
            return Ok(Step::End);
        }
        let next = lists.next;
        let (end, elements) = match elements_inlet {
            None => {
                vortex_ensure!(
                    lists.offsets[lists.len()] == lists.offsets[next],
                    "Lists hold elements but none were read"
                );
                (
                    lists.len(),
                    Canonical::empty(&self.elements_dtype).into_array(),
                )
            }
            Some(index) => {
                let mut inlet = cx.inlet(index);
                let available = lists.cursor + inlet.rows() as u64;
                // The lists whose elements have all arrived.
                let mut end = next;
                while end < lists.len() && lists.end(end) <= available {
                    end += 1;
                }
                vortex_ensure!(
                    end > next,
                    "List elements ended before list {next} of {}",
                    lists.len()
                );
                let first = lists.starts[next];
                vortex_ensure!(first >= lists.cursor, "List elements were dropped");
                let stop = (next..end)
                    .map(|list| lists.end(list))
                    .max()
                    .unwrap_or(first);
                let taken = usize::try_from(stop - lists.cursor)?;
                let batch = if taken == 0 {
                    Canonical::empty(&self.elements_dtype).into_array()
                } else {
                    take_rows(&mut inlet, taken)?
                };
                let selected = usize::try_from(lists.offsets[end] - lists.offsets[next])?;
                let elements = if selected == taken {
                    batch
                } else {
                    let mut slices: Vec<(usize, usize)> = Vec::new();
                    for list in next..end {
                        let start = usize::try_from(lists.starts[list] - lists.cursor)?;
                        let stop = usize::try_from(lists.end(list) - lists.cursor)?;
                        if start == stop {
                            continue;
                        }
                        match slices.last_mut() {
                            Some(last) if last.1 == start => last.1 = stop,
                            _ => slices.push((start, stop)),
                        }
                    }
                    batch.filter(Mask::from_slices(taken, slices))?
                };
                lists.cursor = stop;
                (end, elements)
            }
        };
        let base = lists.offsets[next];
        let offsets: Buffer<u64> = lists.offsets[next..=end]
            .iter()
            .map(|&offset| offset - base)
            .collect();
        let validity = match &lists.validity {
            Some(validity) => Validity::Array(validity.slice(next..end)?),
            None => Validity::NonNullable,
        };
        lists.next = end;
        // SAFETY: the offsets start at zero, never decrease, and end at the number of elements
        // kept, and the validity holds one value per list.
        let list = unsafe { ListArray::new_unchecked(elements, offsets.into_array(), validity) };
        Ok(Step::Final(list.into_array()))
    }
}
