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

/// The dtype list offsets are decoded to.
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
/// inlet holds the element rows `read`, which the list's element index bounded the selected
/// lists' elements with when the source was compiled, or every element of the column for a list
/// without one. The source keeps of them the elements the offsets name.
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
