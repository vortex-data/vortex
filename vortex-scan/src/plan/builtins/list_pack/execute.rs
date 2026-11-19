// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists once their offsets, and then their elements, have been read.

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
use vortex_buffer::BufferMut;
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
use crate::execution::operators::UNBOUNDED;
use crate::execution::operators::join_inlet;
use crate::plan::PlanRef;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

/// Assembles the lists a mask selects in two rounds of reads.
///
/// Its inlets hold the offsets from the first selected list to one past the last, and, when
/// nullable, the validity of the selected lists. Once both are joined, it compiles the elements
/// of the selected non-null lists, exactly, through [`Cx::compile`], and emits every list once
/// those elements have all arrived.
///
/// The elements of null lists are not read: a null list comes out empty.
pub(crate) struct ListSource {
    elements: PlanRef,
    offsets_dtype: DType,
    nullability: Nullability,
    /// The lists to produce, over the rows the offsets cover.
    mask: Mask,
    phase: Phase,
}

enum Phase {
    /// Waiting for the offsets and validity to close.
    Joining,
    /// Waiting for the elements, compiled into this inlet, to close.
    Elements {
        inlet: usize,
        lists: Lists,
    },
    /// Every list is ready to emit, with no element to wait for.
    Empty(Lists),
    Done,
}

/// The selected lists, once their offsets and validity have been joined.
struct Lists {
    /// Where each selected list starts among the selected elements, and the end of the last, in
    /// the offsets' stored type.
    offsets: ArrayRef,
    /// The number of selected elements.
    elements: u64,
    validity: Validity,
}

impl ListSource {
    pub(crate) fn new(
        elements: PlanRef,
        offsets_dtype: DType,
        nullability: Nullability,
        mask: Mask,
    ) -> Self {
        Self {
            elements,
            offsets_dtype,
            nullability,
            mask,
            phase: Phase::Joining,
        }
    }

    /// Joins the offsets and validity, and works out the elements the selected non-null lists
    /// hold: their output offsets, and the mask of those elements over the element rows they
    /// span. The mask is `None` when they hold no element.
    fn plan(&self, cx: &mut Cx<'_>) -> VortexResult<(Lists, Option<(u64, Mask)>)> {
        let raw = join_inlet(cx, OFFSETS, &self.offsets_dtype)?
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(cx.exec())?;
        let raw = raw.as_slice::<u64>();
        vortex_ensure_eq!(raw.len(), self.mask.len() + 1);
        vortex_ensure!(raw.is_sorted(), "List offsets decrease");
        let validity = if self.nullability.is_nullable() {
            Some(join_inlet(
                cx,
                VALIDITY,
                &DType::Bool(Nullability::NonNullable),
            )?)
        } else {
            None
        };
        let valid = match &validity {
            Some(validity) => {
                vortex_ensure_eq!(validity.len(), self.mask.true_count());
                validity.clone().execute::<Mask>(cx.exec())?
            }
            None => Mask::new_true(self.mask.true_count()),
        };

        let mut offsets = BufferMut::<u64>::with_capacity(self.mask.true_count() + 1);
        offsets.push(0);
        let mut total = 0u64;
        // Element rows the kept lists hold, merged where they touch.
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut valid = valid.iter();
        let mut select = |start: usize, end: usize| {
            for (window, is_valid) in raw[start..=end].windows(2).zip(&mut valid) {
                let (first, last) = (window[0], window[1]);
                if is_valid && first < last {
                    total += last - first;
                    match ranges.last_mut() {
                        Some(range) if range.1 == first => range.1 = last,
                        _ => ranges.push((first, last)),
                    }
                }
                offsets.push(total);
            }
        };
        match self.mask.slices() {
            AllOr::All => select(0, self.mask.len()),
            AllOr::None => {}
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    select(start, end);
                }
            }
        }

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
        let lists = Lists {
            offsets,
            elements: total,
            validity: validity.map_or(Validity::NonNullable, Validity::Array),
        };
        let (Some(&(start, _)), Some(&(_, end))) = (ranges.first(), ranges.last()) else {
            return Ok((lists, None));
        };
        let len = usize::try_from(end - start)?;
        let mask = if ranges.len() == 1 {
            Mask::new_true(len)
        } else {
            Mask::from_slices(
                len,
                ranges
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
        Ok((lists, Some((start, mask))))
    }

    /// The lists, over `elements`, the selected elements in order.
    fn emit(&self, lists: Lists, elements: ArrayRef) -> VortexResult<Step> {
        vortex_ensure_eq!(elements.len() as u64, lists.elements);
        // SAFETY: the offsets start at zero, never decrease, and end at the number of selected
        // elements, which `elements` holds; the validity holds one value per selected list.
        let list = unsafe { ListArray::new_unchecked(elements, lists.offsets, lists.validity) };
        Ok(Step::Final(list.into_array()))
    }
}

impl Source for ListSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        match &self.phase {
            Phase::Joining => {
                let joined = 1 + usize::from(self.nullability.is_nullable());
                for index in 0..joined {
                    if !cx.inlet(index).closed() {
                        return Ok(SourceReady::Blocked(SourceBlocked::Inlet(index)));
                    }
                }
                let (lists, elements) = self.plan(cx)?;
                let Some((start, mask)) = elements else {
                    self.phase = Phase::Empty(lists);
                    return Ok(SourceReady::Ready);
                };
                let rows = start..start + mask.len() as u64;
                // The lists are emitted together, once all their elements are here: emitting a
                // batch per element segment would cut the struct a list is a field of at every
                // one of its segments' edges.
                let inlet = cx.compile(&self.elements, rows, mask, UNBOUNDED);
                self.phase = Phase::Elements { inlet, lists };
                Ok(SourceReady::Blocked(SourceBlocked::Inlet(inlet)))
            }
            // A multi-inlet sink may close its outlet in a run after its last push, so the
            // elements are complete only once the inlet closes.
            Phase::Elements { inlet, .. } => Ok(if cx.inlet(*inlet).closed() {
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
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Elements { inlet, lists } => {
                let elements = join_inlet(cx, inlet, self.elements.dtype())?;
                self.emit(lists, elements)
            }
            Phase::Empty(lists) => {
                self.emit(lists, Canonical::empty(self.elements.dtype()).into_array())
            }
            Phase::Done => Ok(Step::End),
            Phase::Joining => Err(vortex_err!("List computed before its offsets were joined")),
        }
    }
}
