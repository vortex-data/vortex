// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from their offsets, validity and elements.

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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::join_inlet;

/// The dtype list offsets and list indexes are decoded to.
pub(crate) fn wide_offsets_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

/// The offsets inlet `inlet` holds, as `u64`: those of `lists` lists, and one more.
pub(crate) fn stored_offsets(
    cx: &mut Cx<'_>,
    inlet: usize,
    dtype: &DType,
    lists: usize,
) -> VortexResult<Buffer<u64>> {
    let stored = join_inlet(cx, inlet, dtype)?
        .cast(wide_offsets_dtype())?
        .execute::<PrimitiveArray>(cx.exec())?;
    vortex_ensure_eq!(stored.len(), lists + 1);
    vortex_ensure!(
        stored.as_slice::<u64>().is_sorted(),
        "List offsets decrease"
    );
    Ok(stored.to_buffer::<u64>())
}

/// The element rows the lists `mask` selects hold, merged where they touch, in order.
///
/// `stored` holds the offsets of the lists `mask` covers, and one more.
fn element_spans(stored: &[u64], mask: &Mask) -> Vec<(u64, u64)> {
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut add = |start: usize, end: usize| {
        let (first, last) = (stored[start], stored[end]);
        if first < last {
            match spans.last_mut() {
                Some(span) if span.1 == first => span.1 = last,
                _ => spans.push((first, last)),
            }
        }
    };
    match mask.slices() {
        AllOr::All => add(0, mask.len()),
        AllOr::None => {}
        AllOr::Some(slices) => {
            for &(start, end) in slices {
                add(start, end);
            }
        }
    }
    spans
}

/// The element rows spanning `spans`, and which of them the spans hold. `None` when the spans
/// hold no element.
pub(crate) fn element_selection(spans: &[(u64, u64)]) -> VortexResult<Option<(Range<u64>, Mask)>> {
    let (Some(&(start, _)), Some(&(_, end))) = (spans.first(), spans.last()) else {
        return Ok(None);
    };
    let len = usize::try_from(end - start)?;
    let mask = if spans.len() == 1 {
        Mask::new_true(len)
    } else {
        Mask::from_slices(
            len,
            spans
                .iter()
                .map(|&(first, last)| {
                    Ok((
                        usize::try_from(first - start)?,
                        usize::try_from(last - start)?,
                    ))
                })
                .collect::<VortexResult<Vec<_>>>()?,
        )
    };
    Ok(Some((start..end, mask)))
}

/// A coarse copy of a list column's offsets, kept with the list's plan.
///
/// Entry `i` says list `lists[i]` starts at element row `offsets[i]`. Both increase, so
/// consecutive entries bound a window of whole lists, and the elements of a run of lists lie
/// between the entry at or before its first list and the entry at or after its end.
///
/// The index is what lets a chunk be compiled before it has read anything: its lookups are two
/// binary searches over plain integers. The offsets themselves cannot play that part. They are
/// data, read with the chunk, and stored however they compress best, which for increasing
/// integers is as deltas: reading one value of those decodes a block of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListIndex {
    lists: Buffer<u64>,
    offsets: Buffer<u64>,
}

impl ListIndex {
    /// The index whose entry `i` says list `lists[i]` starts at element row `offsets[i]`.
    ///
    /// The lists must strictly increase and the offsets must not decrease.
    pub fn try_new(lists: Buffer<u64>, offsets: Buffer<u64>) -> VortexResult<Self> {
        vortex_ensure_eq!(lists.len(), offsets.len());
        vortex_ensure!(
            lists.windows(2).all(|pair| pair[0] < pair[1]) && offsets.is_sorted(),
            "A list index must increase"
        );
        Ok(Self { lists, offsets })
    }

    /// The list each window but the first starts at.
    pub fn lists(&self) -> &[u64] {
        &self.lists
    }

    /// The element row each of [`lists`](Self::lists) starts at.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// The element rows holding lists `lists` of a column of `elements` elements, to within a
    /// window on either side.
    fn window(&self, lists: Range<u64>, elements: u64) -> (u64, u64) {
        let before = self.lists.partition_point(|&list| list <= lists.start);
        let start = before
            .checked_sub(1)
            .map_or(0, |entry| self.offsets[entry]);
        let after = self.lists.partition_point(|&list| list < lists.end);
        let end = self.offsets.get(after).copied().unwrap_or(elements);
        (start, end)
    }

    /// The element rows to read for the lists `mask` selects of `rows`, merged where their
    /// windows meet, in order.
    pub(crate) fn windows(&self, rows: &Range<u64>, mask: &Mask, elements: u64) -> Vec<(u64, u64)> {
        let mut spans: Vec<(u64, u64)> = Vec::new();
        let mut add = |start: usize, end: usize| {
            let lists = rows.start + start as u64..rows.start + end as u64;
            let (first, last) = self.window(lists, elements);
            if first < last {
                match spans.last_mut() {
                    Some(span) if span.1 >= first => span.1 = span.1.max(last),
                    _ => spans.push((first, last)),
                }
            }
        };
        match mask.slices() {
            AllOr::All => add(0, mask.len()),
            AllOr::None => {}
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    add(start, end);
                }
            }
        }
        spans
    }
}

/// Builds a list column's index from its offsets, as they are written.
///
/// The index samples the offsets at list boundaries, a window of elements apart: a window is as
/// many whole lists as hold at most `window` elements, or one list that holds more. Bounding a
/// selection of lists with it reads under a window of elements too many on either side, whatever
/// the lists' lengths, and lists longer than a window are found exactly.
///
/// No one window suits every column. Short lists under a selective filter want a small one, so
/// little is read around the lists that survive. Long lists want a large one, or every list is
/// a window and the index is as long as the offsets. So the window starts small and doubles
/// whenever the index would hold more than `max_entries` entries: a column gets the finest index
/// that fits, and the index stays small enough to read whole before a scan's first chunk.
pub struct ListIndexBuilder {
    window: u64,
    max_entries: usize,
    /// The lists whose offsets have been pushed.
    list_count: u64,
    /// Where the first list starts.
    first: u64,
    /// Where the next list starts: the last offset pushed.
    next: Option<u64>,
    /// Where the window being filled starts, and the lists it holds so far.
    window_start: u64,
    window_lists: u64,
    lists: Vec<u64>,
    offsets: Vec<u64>,
}

impl ListIndexBuilder {
    /// A builder of an index of at most `max_entries` entries, whose windows hold at most
    /// `window` elements, or more where that many entries would not hold the column.
    pub fn new(window: u64, max_entries: usize) -> Self {
        Self {
            window: window.max(1),
            max_entries,
            list_count: 0,
            first: 0,
            next: None,
            window_start: 0,
            window_lists: 0,
            lists: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// Adds the next offsets of the column. The first offset ever pushed is where the first list
    /// starts; every other is where a list ends.
    pub fn push(&mut self, offsets: &[u64]) {
        for &end in offsets {
            let Some(start) = self.next.replace(end) else {
                self.first = end;
                self.window_start = end;
                continue;
            };
            if self.window_lists > 0 && end - self.window_start > self.window {
                self.lists.push(self.list_count);
                self.offsets.push(start);
                self.window_start = start;
                self.window_lists = 0;
            }
            self.window_lists += 1;
            self.list_count += 1;
            while self.lists.len() > self.max_entries {
                self.widen(end);
            }
        }
    }

    /// Doubles the window and packs the windows so far into windows of the new size, in order.
    /// `end` is where the window being filled ends so far.
    ///
    /// A new window is as many whole old windows as hold at most the new size, or one old window
    /// that holds more, which is one list: what the index promises of a window still holds.
    fn widen(&mut self, end: u64) {
        self.window = self.window.saturating_mul(2);
        let entries = self.lists.len();
        let mut lists = Vec::with_capacity(entries / 2 + 1);
        let mut offsets = Vec::with_capacity(entries / 2 + 1);
        // Where the new window being packed starts, and whether it holds an old window yet.
        let mut start = self.first;
        let mut holds = false;
        // Old window `index` starts at entry `index - 1`, or at the first list, and ends where
        // the next starts, or at `end` for the window being filled.
        for index in 0..=entries {
            let old_start = index
                .checked_sub(1)
                .map_or(self.first, |entry| self.offsets[entry]);
            let old_end = self.offsets.get(index).copied().unwrap_or(end);
            if holds && old_end - start > self.window {
                lists.push(self.lists[index - 1]);
                offsets.push(old_start);
                start = old_start;
            }
            holds = true;
        }
        // The window being filled now starts where the last new window does. It holds a list,
        // since it holds the old window being filled.
        self.window_start = start;
        self.lists = lists;
        self.offsets = offsets;
    }

    /// The lists whose offsets have been pushed.
    pub fn list_count(&self) -> u64 {
        self.list_count
    }

    /// The index, or `None` for a column of one window, which the index would say nothing of.
    pub fn finish(self) -> Option<ListIndex> {
        (!self.lists.is_empty()).then(|| ListIndex {
            lists: Buffer::from(self.lists),
            offsets: Buffer::from(self.offsets),
        })
    }
}

/// Which of the element rows `read` holds, in order, are those of `exact`: element rows the
/// spans of `read` cover.
fn within(read: &[(u64, u64)], exact: &[(u64, u64)]) -> VortexResult<Mask> {
    let len = usize::try_from(read.iter().map(|(start, end)| end - start).sum::<u64>())?;
    let mut slices = Vec::with_capacity(exact.len());
    let mut spans = read.iter();
    // The element rows read before the span `exact` has reached, and that span.
    let mut before = 0u64;
    let mut span = spans.next();
    for &(start, end) in exact {
        while let Some(&(_, span_end)) = span
            && span_end <= start
        {
            let &(span_start, _) = span.vortex_expect("checked above");
            before += span_end - span_start;
            span = spans.next();
        }
        let &(span_start, span_end) =
            span.ok_or_else(|| vortex_err!("List elements {start}..{end} were not read"))?;
        vortex_ensure!(
            span_start <= start && end <= span_end,
            "List elements {start}..{end} were not read whole"
        );
        slices.push((
            usize::try_from(before + start - span_start)?,
            usize::try_from(before + end - span_start)?,
        ));
    }
    Ok(Mask::from_slices(len, slices))
}

/// Where each selected list starts among the selected elements, and the end of the last.
fn selected_offsets(stored: &[u64], mask: &Mask) -> BufferMut<u64> {
    let mut offsets = BufferMut::<u64>::with_capacity(mask.true_count() + 1);
    match mask.slices() {
        AllOr::All => {
            let base = stored[0];
            offsets.extend(stored.iter().map(|offset| offset - base));
        }
        AllOr::None => offsets.push(0),
        AllOr::Some(slices) => {
            let mut total = 0u64;
            offsets.push(0);
            for &(start, end) in slices {
                for window in stored[start..=end].windows(2) {
                    total += window[1] - window[0];
                    offsets.push(total);
                }
            }
        }
    }
    offsets
}

/// The inlet a [`ListSource`] reads its lists' offsets from.
pub(crate) const OFFSETS_INLET: usize = 0;

/// Assembles the lists a mask selects, once every inlet has closed.
///
/// Its first inlet holds the offsets of the lists the mask covers, and one more. Its elements
/// inlet holds the element rows `read`, which the list's index bounded the selected lists'
/// elements with when the source was compiled, or every element of the column for a list without
/// one. The source keeps of them the elements the offsets name.
///
/// The lists are emitted together: emitting a batch per element segment would cut the struct a
/// list is a field of at every one of its segments' edges.
pub(crate) struct ListSource {
    elements_dtype: DType,
    offsets_dtype: DType,
    /// The element rows the elements inlet holds, in order.
    read: Vec<(u64, u64)>,
    validity: Option<usize>,
    elements: Option<usize>,
    inlets: usize,
    /// The lists to produce, over the rows the offsets cover.
    mask: Mask,
    done: bool,
}

impl ListSource {
    pub(crate) fn new(
        elements_dtype: DType,
        offsets_dtype: DType,
        read: Vec<(u64, u64)>,
        validity: Option<usize>,
        elements: Option<usize>,
        inlets: usize,
        mask: Mask,
    ) -> Self {
        Self {
            elements_dtype,
            offsets_dtype,
            read,
            validity,
            elements,
            inlets,
            mask,
            done: false,
        }
    }

    fn assemble(&mut self, cx: &mut Cx<'_>) -> VortexResult<ArrayRef> {
        let stored = stored_offsets(cx, OFFSETS_INLET, &self.offsets_dtype, self.mask.len())?;
        let exact = element_spans(&stored, &self.mask);
        let elements = match self.elements {
            _ if exact.is_empty() => Canonical::empty(&self.elements_dtype).into_array(),
            Some(inlet) => {
                let read = join_inlet(cx, inlet, &self.elements_dtype)?;
                if exact == self.read {
                    read
                } else {
                    read.filter(within(&self.read, &exact)?)?
                }
            }
            None => return Err(vortex_err!("List elements {exact:?} were not read")),
        };
        let offsets = selected_offsets(&stored, &self.mask);
        vortex_ensure_eq!(
            offsets.last().copied().unwrap_or(0),
            elements.len() as u64,
            "List elements do not match the selected offsets"
        );
        let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable).into_array();
        let offsets = if self.offsets_dtype.as_ptype() == PType::U64 {
            offsets
        } else {
            // The selected elements are no more than the elements the stored offsets span, so
            // they fit the stored type.
            offsets
                .cast(self.offsets_dtype.clone())?
                .execute::<PrimitiveArray>(cx.exec())?
                .into_array()
        };
        let validity = match self.validity {
            Some(inlet) => Validity::Array(join_inlet(
                cx,
                inlet,
                &DType::Bool(Nullability::NonNullable),
            )?),
            None => Validity::NonNullable,
        };
        // SAFETY: the offsets start at zero, never decrease, and end at the number of selected
        // elements, which `elements` holds; the validity holds one value per selected list.
        Ok(unsafe { ListArray::new_unchecked(elements, offsets, validity) }.into_array())
    }
}

impl Source for ListSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        // A multi-inlet sink may close its outlet in a run after its last push, so an inlet is
        // complete only once it closes.
        for index in 0..self.inlets {
            if !cx.inlet(index).closed() {
                return Ok(SourceReady::Blocked(SourceBlocked::Inlet(index)));
            }
        }
        Ok(SourceReady::Ready)
    }
}

impl Operator for ListSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        self.done = true;
        Ok(Step::Final(self.assemble(cx)?))
    }
}
