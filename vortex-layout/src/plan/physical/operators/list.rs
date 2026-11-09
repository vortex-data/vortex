// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling the selected rows of a list from its offsets and only the elements they name.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::UNBOUNDED;
use super::join_inlet;
use crate::plan::physical::ListPackPlan;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

/// What a [`ListSource`] waits for next.
enum Phase {
    /// The offsets, and the validity of a nullable list, of the selected rows.
    Offsets,
    /// The elements the selected rows name, in the inlet with this index, and the offsets of
    /// the assembled list.
    Elements(usize, Buffer<u64>),
    /// No selected row has an element.
    Empty(Buffer<u64>),
    /// The list has been emitted.
    Done,
}

/// The elements that the selected rows of a list name: a range of the elements' domain, and the
/// elements of it that the rows hold.
struct ElementRows {
    rows: Range<u64>,
    mask: Mask,
}

/// Emits the rows `mask` selects of a list.
///
/// Its first inlet holds the offsets that start and end each selected row, and its second the
/// validity of the selected rows when the list is nullable. Once they have closed, it asks for
/// the elements of exactly the selected rows to be compiled into a new inlet, so a sparse
/// selection reads only the elements it returns. It emits the selected rows as one list.
pub(crate) struct ListSource {
    plan: ListPackPlan,
    /// The selected rows, as positions in the rows the source was compiled over.
    mask: Mask,
    phase: Phase,
    validity: Option<ArrayRef>,
}

impl ListSource {
    /// A source of the rows `mask` selects of `plan`.
    pub(crate) fn new(plan: ListPackPlan, mask: Mask) -> Self {
        Self {
            plan,
            mask,
            phase: Phase::Offsets,
            validity: None,
        }
    }

    /// The inlets of the offsets and the validity are taken whole, so they never block.
    pub(crate) fn capacity(_inlet: usize) -> usize {
        UNBOUNDED
    }

    /// The offsets mask of the rows `mask` selects: each selected row needs the offset that
    /// starts it and the one that ends it, so the mask is one longer than `mask`.
    pub(crate) fn offsets_mask(mask: &Mask) -> Mask {
        let mut slices: Vec<(usize, usize)> = Vec::new();
        for index in selected_indices(mask) {
            match slices.last_mut() {
                Some((_, end)) if *end >= index => *end = index + 2,
                _ => slices.push((index, index + 2)),
            }
        }
        Mask::from_slices(mask.len() + 1, slices)
    }

    /// Reads the offsets of the selected rows, and returns the elements they name, as a range
    /// of the elements' domain and a mask over it, and the offsets of the assembled list.
    fn element_rows(&self, cx: &mut Cx<'_>) -> VortexResult<(Option<ElementRows>, Buffer<u64>)> {
        let offsets_plan = self.plan.offsets()?;
        let offsets = join_inlet(cx, OFFSETS, offsets_plan.dtype())?
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Canonical>(cx.exec())?
            .into_primitive();
        let offsets = offsets.as_slice::<u64>();

        // The offsets inlet holds, in order, the offset before and after each selected row,
        // with the offsets that two neighbouring rows share once.
        let positions = selected_indices(&Self::offsets_mask(&self.mask)).collect::<Vec<_>>();
        vortex_ensure_eq!(offsets.len(), positions.len());

        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut compact = Vec::with_capacity(self.mask.true_count() + 1);
        compact.push(0u64);
        let mut next = 0;
        for row in selected_indices(&self.mask) {
            // Both sequences are sorted, so the offset of each row is found by walking forward.
            while positions[next] < row {
                next += 1;
            }
            let (start, end) = (offsets[next], offsets[next + 1]);
            if end < start {
                vortex_bail!("List offsets decrease from {start} to {end}");
            }

            if end > start {
                match ranges.last_mut() {
                    Some((_, last)) if *last == start => *last = end,
                    Some((_, last)) if *last > start => {
                        vortex_bail!("List rows overlap at element {start}")
                    }
                    _ => ranges.push((start, end)),
                }
            }

            let size = end - start;
            let previous = compact.last().copied().unwrap_or_default();
            compact.push(
                previous
                    .checked_add(size)
                    .ok_or_else(|| vortex_err!("List element count overflows u64"))?,
            );
        }

        let compact = Buffer::from(compact);
        let (Some(&(first, _)), Some(&(_, last))) = (ranges.first(), ranges.last()) else {
            return Ok((None, compact));
        };

        let elements = self.plan.elements()?;
        if last > elements.row_count() {
            vortex_bail!(
                "List offsets end at element {last} of {}",
                elements.row_count()
            );
        }

        let len = usize::try_from(last - first)?;
        let slices = ranges
            .iter()
            .map(|&(start, end)| {
                Ok((
                    usize::try_from(start - first)?,
                    usize::try_from(end - first)?,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let rows = ElementRows {
            rows: first..last,
            mask: Mask::from_slices(len, slices),
        };
        Ok((Some(rows), compact))
    }

    /// The selected rows over `elements`, with offsets `offsets`.
    fn assemble(
        &self,
        elements: ArrayRef,
        offsets: Buffer<u64>,
        cx: &mut Cx<'_>,
    ) -> VortexResult<ArrayRef> {
        let validity = match self.validity.clone() {
            Some(validity) => Validity::Array(validity),
            None => Validity::from(self.plan.dtype().nullability()),
        };
        let list = ListArray::try_new(elements, offsets.into_array(), validity, cx.exec())?;
        Ok(list.into_array())
    }
}

/// The positions `mask` sets, in order.
fn selected_indices(mask: &Mask) -> impl Iterator<Item = usize> + '_ {
    (0..mask.len()).filter(move |&index| mask.value(index))
}

impl Source for ListSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        match &self.phase {
            Phase::Offsets => {
                if !cx.inlet(OFFSETS).closed() {
                    return Ok(SourceReady::Blocked(SourceBlocked::Inlet(OFFSETS)));
                }
                if self.plan.validity()?.is_some() && !cx.inlet(VALIDITY).closed() {
                    return Ok(SourceReady::Blocked(SourceBlocked::Inlet(VALIDITY)));
                }

                if let Some(validity) = self.plan.validity()? {
                    self.validity = Some(join_inlet(cx, VALIDITY, validity.dtype())?);
                }

                let (elements, offsets) = self.element_rows(cx)?;
                let Some(ElementRows { rows, mask }) = elements else {
                    self.phase = Phase::Empty(offsets);
                    return Ok(SourceReady::Ready);
                };
                let inlet = cx.compile_inlet(self.plan.elements()?, rows, mask, UNBOUNDED);
                self.phase = Phase::Elements(inlet, offsets);
                Ok(SourceReady::Blocked(SourceBlocked::Inlet(inlet)))
            }
            Phase::Elements(inlet, _) => Ok(if cx.inlet(*inlet).closed() {
                SourceReady::Ready
            } else {
                SourceReady::Blocked(SourceBlocked::Inlet(*inlet))
            }),
            Phase::Empty(_) | Phase::Done => Ok(SourceReady::Ready),
        }
    }
}

impl Operator for ListSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let elements_dtype = self.plan.elements()?.dtype().clone();
        let (elements, offsets) = match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Elements(inlet, offsets) => (join_inlet(cx, inlet, &elements_dtype)?, offsets),
            Phase::Empty(offsets) => (Canonical::empty(&elements_dtype).into_array(), offsets),
            Phase::Done => return Ok(Step::End),
            Phase::Offsets => vortex_bail!("A list source computed before its offsets arrived"),
        };
        Ok(Step::Final(self.assemble(elements, offsets, cx)?))
    }
}
