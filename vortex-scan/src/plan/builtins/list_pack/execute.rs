// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from their offsets, validity and elements.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Primitive;
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
use vortex_session::VortexSession;

use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::join_inlet;

/// The dtype list offsets are held in once read ahead of a scan's chunks.
pub(crate) fn wide_offsets_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

/// The offsets of a list column read ahead, as `u64`.
pub(crate) fn wide_offsets(offsets: &ArrayRef) -> VortexResult<Buffer<u64>> {
    let primitive = offsets
        .as_opt::<Primitive>()
        .ok_or_else(|| vortex_err!("List offsets read ahead must be primitive, got {offsets}"))?;
    vortex_ensure_eq!(primitive.ptype(), PType::U64);
    Ok(primitive.to_buffer::<u64>())
}

/// The offsets of lists `lists` and one more, as `u64`, from a list column's offsets read ahead.
///
/// Offsets read ahead stay as they were stored: those of long, even lists are a few bytes for
/// any number of lists, and a table can have hundreds of list columns. Only the offsets a chunk
/// selects are decoded, when it is compiled.
pub(crate) fn offsets_ahead(
    offsets: &ArrayRef,
    lists: Range<u64>,
    session: &VortexSession,
) -> VortexResult<Buffer<u64>> {
    let rows = usize::try_from(lists.start)?..usize::try_from(lists.end)? + 1;
    vortex_ensure!(
        rows.end <= offsets.len(),
        "List offsets read ahead do not cover lists {lists:?}"
    );
    let mut ctx = session.create_execution_ctx();
    let stored = offsets
        .slice(rows)?
        .cast(wide_offsets_dtype())?
        .execute::<PrimitiveArray>(&mut ctx)?;
    vortex_ensure!(
        stored.as_slice::<u64>().is_sorted(),
        "List offsets decrease"
    );
    Ok(stored.to_buffer::<u64>())
}

/// The element rows the lists `mask` selects hold, merged where they touch, in order.
///
/// `stored` holds the offsets of the lists `mask` covers, and one more.
pub(crate) fn element_spans(stored: &[u64], mask: &Mask) -> Vec<(u64, u64)> {
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

/// A coarse copy of a list column's offsets, read ahead of a scan's chunks when the offsets are
/// too many to be.
///
/// Entry `i` says list `lists[i]` starts at element row `offsets[i]`. Both increase, so
/// consecutive entries bound a window of whole lists, and the elements of a run of lists lie
/// between the entry at or before its first list and the entry at or after its end.
pub(crate) struct ListIndex {
    lists: Buffer<u64>,
    offsets: Buffer<u64>,
}

impl ListIndex {
    /// The index held by `array`: its lists, then the element rows they start at.
    pub(crate) fn try_new(array: &ArrayRef) -> VortexResult<Self> {
        let entries = wide_offsets(array)?;
        vortex_ensure!(
            entries.len().is_multiple_of(2),
            "A list index holds pairs, got {} values",
            entries.len()
        );
        let count = entries.len() / 2;
        Ok(Self {
            lists: entries.slice(0..count),
            offsets: entries.slice(count..2 * count),
        })
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
pub struct ListIndexBuilder {
    window: u64,
    /// The lists whose offsets have been pushed.
    list_count: u64,
    /// Where the next list starts: the last offset pushed.
    next: Option<u64>,
    /// Where the window being filled starts, and the lists it holds so far.
    window_start: u64,
    window_lists: u64,
    lists: Vec<u64>,
    offsets: Vec<u64>,
}

impl ListIndexBuilder {
    /// A builder of an index whose windows hold at most `window` elements, or one list.
    pub fn new(window: u64) -> Self {
        Self {
            window,
            list_count: 0,
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
        }
    }

    /// The lists whose offsets have been pushed.
    pub fn list_count(&self) -> u64 {
        self.list_count
    }

    /// The index: the list each window but the first starts at, then the element rows those
    /// lists start at.
    pub fn finish(mut self) -> Buffer<u64> {
        self.lists.append(&mut self.offsets);
        Buffer::from(self.lists)
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

/// Where a [`ListSource`] finds the offsets of the lists it assembles.
pub(crate) enum Offsets {
    /// Read ahead of the scan's chunks: the offsets of the lists the mask covers, and one more.
    /// The elements inlet then holds exactly the selected lists' elements.
    Ahead(Buffer<u64>),
    /// Read by this inlet, one more than the lists the mask covers. The elements inlet then holds
    /// the element rows the source was compiled to read: a bound on the selected lists' elements
    /// from the list's index, or every element of the column.
    Inlet(usize),
}

/// Assembles the lists a mask selects, once every inlet has closed.
///
/// The lists are emitted together: emitting a batch per element segment would cut the struct a
/// list is a field of at every one of its segments' edges.
pub(crate) struct ListSource {
    elements_dtype: DType,
    offsets_dtype: DType,
    offsets: Offsets,
    /// The element rows the elements inlet holds, in order, when the offsets are read by an
    /// inlet.
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
        offsets: Offsets,
        read: Vec<(u64, u64)>,
        validity: Option<usize>,
        elements: Option<usize>,
        inlets: usize,
        mask: Mask,
    ) -> Self {
        Self {
            elements_dtype,
            offsets_dtype,
            offsets,
            read,
            validity,
            elements,
            inlets,
            mask,
            done: false,
        }
    }

    fn assemble(&mut self, cx: &mut Cx<'_>) -> VortexResult<ArrayRef> {
        let elements = match self.elements {
            Some(inlet) => join_inlet(cx, inlet, &self.elements_dtype)?,
            None => Canonical::empty(&self.elements_dtype).into_array(),
        };
        let (offsets, elements) = match &self.offsets {
            Offsets::Ahead(stored) => (selected_offsets(stored, &self.mask), elements),
            Offsets::Inlet(inlet) => {
                let stored = join_inlet(cx, *inlet, &self.offsets_dtype)?
                    .cast(wide_offsets_dtype())?
                    .execute::<PrimitiveArray>(cx.exec())?;
                let stored = stored.as_slice::<u64>();
                vortex_ensure_eq!(stored.len(), self.mask.len() + 1);
                vortex_ensure!(stored.is_sorted(), "List offsets decrease");
                let exact = element_spans(stored, &self.mask);
                let elements = if exact.is_empty() {
                    Canonical::empty(&self.elements_dtype).into_array()
                } else if exact == self.read {
                    elements
                } else {
                    elements.filter(within(&self.read, &exact)?)?
                };
                (selected_offsets(stored, &self.mask), elements)
            }
        };
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
