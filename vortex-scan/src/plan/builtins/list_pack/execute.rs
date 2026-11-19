// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists, reading their elements once their offsets give the element rows.

use std::ops::Range;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::join;
use crate::execution::operators::join_inlet;
use crate::execution::pipeline::ReadId;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::execution::pipeline::Turn;
use crate::execution::pipeline::ValueCache;
use crate::plan::ListPackPlan;
use crate::source::SegmentId;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

/// Assembles the selected lists of a row range from its inlets, the offsets of the range and the
/// validity of its selected lists, and from the elements those lists hold.
///
/// The element rows of a list are known only once its offsets are read, so the elements cannot be
/// compiled with the lists. Once both inlets have closed, the source compiles the elements plan
/// over the elements of the selected valid lists as a scan of its own, which it runs within its
/// readiness checks: the reads that scan asks for are this source's reads, and their bytes go
/// back to it. The lists are emitted as one batch once every element has arrived.
pub(crate) struct ListSource {
    plan: ListPackPlan,
    /// The selected lists of the row range the offsets inlet covers.
    selected: Mask,
    session: VortexSession,
    values: ValueCache,
    state: State,
}

enum State {
    /// Waiting for the offsets and the validity.
    Lists,
    /// Reading the elements of the selected valid lists.
    Elements(Box<ElementScan>),
    /// The lists, built and not yet emitted.
    Built(ArrayRef),
    Done,
}

/// The scan of the elements of the selected valid lists.
struct ElementScan {
    scan: Scan,
    /// The scan's reads that have not arrived, by segment, oldest first.
    reads: FxHashMap<SegmentId, SmallVec<[ReadId; 1]>>,
    /// What the scan produced, in element order.
    elements: Vec<ArrayRef>,
    dtype: DType,
    lists: CompactedLists,
}

impl ListSource {
    /// A source of the `selected` lists of a row range, whose offsets inlet holds the offsets of
    /// every list of the range, one more than `selected` is long, and whose validity inlet, present
    /// exactly when the list is nullable, holds the validity of the selected lists.
    pub(crate) fn new(
        plan: ListPackPlan,
        selected: Mask,
        session: VortexSession,
        values: ValueCache,
    ) -> Self {
        Self {
            plan,
            selected,
            session,
            values,
            state: State::Lists,
        }
    }

    /// Compacts the lists once their offsets and validity have arrived, and starts the scan of
    /// their elements.
    fn read_lists(&mut self, cx: &mut Cx<'_>) -> VortexResult<State> {
        let offsets_dtype = self.plan.offsets()?.dtype().clone();
        let offsets = join_inlet(cx, OFFSETS, &offsets_dtype)?
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(cx.exec())?;
        let validity = if self.plan.dtype().is_nullable() {
            let validity = join_inlet(cx, VALIDITY, &DType::Bool(Nullability::NonNullable))?;
            let valid = validity.clone().execute::<Mask>(cx.exec())?;
            Some((validity, valid))
        } else {
            None
        };
        let elements_plan = self.plan.elements()?;
        let lists = CompactedLists::try_new(
            offsets.as_slice::<u64>(),
            &self.selected,
            validity,
            elements_plan.row_count(),
        )?;
        if lists.elements_mask.all_false() {
            let elements = Canonical::empty(elements_plan.dtype()).into_array();
            return Ok(State::Built(lists.build(elements)?));
        }
        let split = Split {
            rows: lists.elements_range.clone(),
            mask: lists.elements_mask.clone(),
        };
        let dtype = elements_plan.dtype().clone();
        let scan = Scan::try_new(self.session.clone(), elements_plan, vec![split])?
            .with_value_cache(self.values.clone());
        Ok(State::Elements(Box::new(ElementScan {
            scan,
            reads: FxHashMap::default(),
            elements: Vec::new(),
            dtype,
            lists,
        })))
    }
}

impl ElementScan {
    /// Hands the bytes that have arrived to the scan and runs it until it waits for more or has
    /// finished, returning the lists once it has.
    fn advance(&mut self, cx: &mut Cx<'_>) -> VortexResult<Option<ArrayRef>> {
        while let Some((segment, bytes)) = cx.take_segment() {
            let read = self
                .reads
                .get_mut(&segment)
                .filter(|reads| !reads.is_empty())
                .map(|reads| reads.remove(0))
                .ok_or_else(|| vortex_err!("List elements received {segment}, never asked for"))?;
            self.scan.deliver(read, bytes)?;
        }
        loop {
            match self.scan.step()? {
                Turn::Read(read) => {
                    cx.request_read(read.segment_id);
                    self.reads.entry(read.segment_id).or_default().push(read.id);
                }
                Turn::Output(_, elements) => self.elements.push(elements),
                Turn::Waiting => return Ok(None),
                Turn::Done => {
                    let elements = join(&self.dtype, std::mem::take(&mut self.elements))?;
                    return self.lists.build(elements).map(Some);
                }
            }
        }
    }
}

impl Source for ListSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        loop {
            self.state = match &mut self.state {
                State::Lists => {
                    if !cx.inlet(OFFSETS).closed() {
                        return Ok(SourceReady::Blocked(SourceBlocked::Inlet(OFFSETS)));
                    }
                    if self.plan.dtype().is_nullable() && !cx.inlet(VALIDITY).closed() {
                        return Ok(SourceReady::Blocked(SourceBlocked::Inlet(VALIDITY)));
                    }
                    self.read_lists(cx)?
                }
                State::Elements(scan) => match scan.advance(cx)? {
                    Some(lists) => State::Built(lists),
                    None => return Ok(SourceReady::Blocked(SourceBlocked::Io)),
                },
                State::Built(_) | State::Done => return Ok(SourceReady::Ready),
            };
        }
    }
}

impl Operator for ListSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match std::mem::replace(&mut self.state, State::Done) {
            State::Built(lists) => Step::Final(lists),
            State::Done => Step::End,
            State::Lists | State::Elements(_) => {
                vortex_bail!("List source computed before its lists were ready")
            }
        })
    }
}

/// The selected lists of a row range, with only the elements of the valid ones.
///
/// `list_transform` runs a lambda that can fail only on the elements of valid lists, and an
/// expression over the elements may be such a lambda's body, so the elements of a null or
/// unselected list are never read.
struct CompactedLists {
    /// The element rows from the first list of the range to its last.
    elements_range: Range<u64>,
    /// The elements of the selected valid lists, relative to `elements_range`.
    elements_mask: Mask,
    /// The offsets of the selected lists into those elements. A null list is empty.
    offsets: PrimitiveArray,
    validity: Validity,
}

impl CompactedLists {
    /// Compacts the `selected` lists of a range whose lists start at `offsets`, one more offset than
    /// `selected` is long, into an elements child of `elements_row_count` rows. `validity` holds
    /// the validity of the selected lists of a nullable list, as an array and as a mask.
    fn try_new(
        offsets: &[u64],
        selected: &Mask,
        validity: Option<(ArrayRef, Mask)>,
        elements_row_count: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offsets.len() == selected.len() + 1,
            "{} list offsets cannot bound {} lists",
            offsets.len(),
            selected.len()
        );
        let first = offsets[0];
        let last = offsets[selected.len()];
        vortex_ensure!(
            first <= last && last <= elements_row_count,
            "List offsets {first}..{last} exceed the {elements_row_count} elements"
        );
        let mut slices: Vec<(usize, usize)> = Vec::new();
        let mut compacted = BufferMut::<u64>::with_capacity(selected.true_count() + 1);
        let mut kept = 0;
        compacted.push(kept);
        let mut valid = validity.as_ref().map(|(_, valid)| valid.iter());
        for (row, is_selected) in selected.iter().enumerate() {
            if !is_selected {
                continue;
            }
            let is_valid = match valid.as_mut() {
                Some(valid) => valid
                    .next()
                    .ok_or_else(|| vortex_err!("List validity is shorter than the selection"))?,
                None => true,
            };
            let (start, end) = (offsets[row], offsets[row + 1]);
            vortex_ensure!(
                first <= start && start <= end && end <= last,
                "List offsets {start}..{end} are out of order"
            );
            if is_valid && start < end {
                kept += end - start;
                let slice = (
                    usize::try_from(start - first)?,
                    usize::try_from(end - first)?,
                );
                match slices.last_mut() {
                    Some((_, previous_end)) if *previous_end == slice.0 => *previous_end = slice.1,
                    _ => slices.push(slice),
                }
            }
            compacted.push(kept);
        }
        let len = usize::try_from(last - first)?;
        let elements_mask = match slices.as_slice() {
            [] => Mask::new_false(len),
            [(0, end)] if *end == len => Mask::new_true(len),
            _ => Mask::from_slices(len, slices),
        };
        Ok(Self {
            elements_range: first..last,
            elements_mask,
            offsets: PrimitiveArray::new(compacted.freeze(), Validity::NonNullable),
            validity: validity.map_or(Validity::NonNullable, |(validity, _)| {
                Validity::Array(validity)
            }),
        })
    }

    /// The lists over `elements`, the elements the mask keeps.
    fn build(&self, elements: ArrayRef) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            elements.len() == self.elements_mask.true_count(),
            "List elements produced {} rows for {} elements",
            elements.len(),
            self.elements_mask.true_count()
        );
        // SAFETY: the compacted offsets start at zero, never decrease and end at the number of
        // elements the mask keeps, which the elements hold, and the validity covers one value per
        // selected list.
        let offsets = self.offsets.clone().into_array();
        Ok(
            unsafe { ListArray::new_unchecked(elements, offsets, self.validity.clone()) }
                .into_array(),
        )
    }
}
