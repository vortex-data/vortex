// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The sources and stages plans compile to.
//!
//! Every operator does work proportional to the rows it is handed: no operator looks at a batch
//! twice, holds more than the batches it must join, or scans its inlets beyond the ones it reads.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Shared;
use vortex_array::arrays::SharedArray;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::operators::Operator as BinaryOperator;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Blocked;
use super::Cx;
use super::Inlet;
use super::Input;
use super::Operator;
use super::Source;
use super::Step;
use crate::plan::ListPackPlan;
use crate::plan::SegmentScanPlan;
use crate::plan::TakePlan;
use crate::segments::SegmentId;

/// The selected fraction of a predicate's batch at or above which it is executed whole and then
/// filtered, as the default scan's flat reader does.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// An inlet capacity that never blocks the writer, for a reader that needs its inlet whole.
const UNBOUNDED: usize = usize::MAX;

/// Keeps the rows of `array` that `mask` selects. `predicate` says the array is a predicate's
/// result, which, mostly selected, is cheaper to execute whole than to filter lazily.
pub(crate) fn keep_selected(
    array: ArrayRef,
    mask: Mask,
    predicate: bool,
    cx: &mut Cx<'_>,
) -> VortexResult<ArrayRef> {
    if mask.all_true() {
        return Ok(array);
    }
    if predicate && mask.density() >= EXPR_EVAL_THRESHOLD {
        return array
            .execute::<Canonical>(cx.exec())?
            .into_array()
            .filter(mask);
    }
    array.filter(mask)
}

/// Decodes the whole of a scan's segment.
pub(crate) fn decode(
    plan: &SegmentScanPlan,
    segment: vortex_array::buffer::BufferHandle,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let serialized = match plan.array_tree() {
        Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?,
        None => SerializedArray::try_from(segment)?,
    };
    serialized.decode(
        plan.dtype(),
        usize::try_from(plan.row_count())?,
        plan.array_ctx(),
        session,
    )
}

/// Slices a whole decoded segment to `slice`, then keeps the rows `filter` selects.
fn select(
    array: ArrayRef,
    slice: Option<&Range<usize>>,
    filter: Option<&Mask>,
) -> VortexResult<ArrayRef> {
    let array = match slice {
        Some(slice) => array.slice(slice.clone())?,
        None => array,
    };
    match filter {
        Some(filter) => array.filter(filter.clone()),
        None => Ok(array),
    }
}

/// Takes the first `len` rows of the inlet's front batch: the batch itself when it is that
/// long, otherwise a slice, leaving the rest in place.
fn take_rows(inlet: &mut Inlet<'_>, len: usize) -> VortexResult<ArrayRef> {
    let front = inlet
        .peek_mut()
        .ok_or_else(|| vortex_err!("Taking rows from an empty inlet"))?;
    if front.len() == len {
        return inlet
            .take()
            .ok_or_else(|| vortex_err!("Taking rows from an empty inlet"));
    }
    let rest = front.slice(len..front.len())?;
    std::mem::replace(front, rest).slice(0..len)
}

/// Joins arrays covering consecutive rows into one.
pub(crate) fn join(dtype: &DType, mut arrays: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    match arrays.len() {
        0 => Ok(Canonical::empty(dtype).into_array()),
        1 => Ok(arrays.remove(0)),
        _ => Ok(vortex_array::arrays::ChunkedArray::try_new(arrays, dtype.clone())?.into_array()),
    }
}

/// Takes every batch of a closed inlet.
fn drain(inlet: &mut Inlet<'_>) -> Vec<ArrayRef> {
    let mut batches = Vec::with_capacity(inlet.len());
    while let Some(batch) = inlet.take() {
        batches.push(batch);
    }
    batches
}

enum ScanState {
    Request,
    Waiting,
    Done,
}

/// Reads one segment, decodes it, and emits its rows of `slice`, filtered by `filter`.
pub(crate) struct ScanSource {
    plan: SegmentScanPlan,
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
    state: ScanState,
}

impl ScanSource {
    pub(crate) fn new(
        plan: SegmentScanPlan,
        slice: Option<Range<usize>>,
        filter: Option<Mask>,
    ) -> Self {
        Self {
            plan,
            slice,
            filter,
            state: ScanState::Request,
        }
    }
}

impl Operator for ScanSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        match self.state {
            ScanState::Request => vortex_bail!("Segment scan computed before its read"),
            ScanState::Waiting => {
                let bytes = cx
                    .take_bytes()
                    .ok_or_else(|| vortex_err!("Segment scan ran without its bytes"))?;
                self.state = ScanState::Done;
                let array = decode(&self.plan, bytes, cx.session())?;
                Ok(Step::Last(select(
                    array,
                    self.slice.as_ref(),
                    self.filter.as_ref(),
                )?))
            }
            ScanState::Done => Ok(Step::Finished),
        }
    }
}

impl Source for ScanSource {
    fn request(&mut self) -> Option<SegmentId> {
        match self.state {
            ScanState::Request => {
                self.state = ScanState::Waiting;
                Some(self.plan.segment_id())
            }
            ScanState::Waiting | ScanState::Done => None,
        }
    }
}

/// Emits its inlet's batches as they arrive.
pub(crate) struct PortSource;

impl Operator for PortSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let mut inlet = cx.inlet(0);
        match inlet.take() {
            Some(batch) if inlet.is_empty() => Ok(Step::Last(batch)),
            Some(batch) => Ok(Step::More(batch)),
            None if inlet.closed() => Ok(Step::Finished),
            None => Ok(Step::Blocked(Blocked::Inlet(0))),
        }
    }
}

impl Source for PortSource {
    fn inlet_count(&self) -> usize {
        1
    }
}

/// Emits one batch it was built with.
pub(crate) struct OnceSource(Option<ArrayRef>);

impl OnceSource {
    pub(crate) fn new(batch: ArrayRef) -> Self {
        Self(Some(batch))
    }
}

impl Operator for OnceSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match self.0.take() {
            Some(batch) => Step::Last(batch),
            None => Step::Finished,
        })
    }
}

impl Source for OnceSource {}

/// Emits its inlets one after another, each to its end.
pub(crate) struct ConcatSource {
    inlets: usize,
    current: usize,
}

impl ConcatSource {
    pub(crate) fn new(inlets: usize) -> Self {
        Self { inlets, current: 0 }
    }
}

impl Operator for ConcatSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        while self.current < self.inlets {
            let mut inlet = cx.inlet(self.current);
            if let Some(batch) = inlet.take() {
                return Ok(if inlet.is_empty() && !inlet.closed() {
                    Step::Last(batch)
                } else {
                    Step::More(batch)
                });
            }
            if !inlet.closed() {
                return Ok(Step::Blocked(Blocked::Inlet(self.current)));
            }
            self.current += 1;
        }
        Ok(Step::Finished)
    }
}

impl Source for ConcatSource {
    fn inlet_count(&self) -> usize {
        self.inlets
    }
}

/// Zips its inlets, one per field and the validity last when nullable, into structs, taking as
/// many rows from each as the shortest front batch holds.
pub(crate) struct PackSource {
    fields: StructFields,
    nullable: bool,
    inlets: usize,
}

impl PackSource {
    pub(crate) fn new(fields: StructFields, nullable: bool, inlets: usize) -> Self {
        Self {
            fields,
            nullable,
            inlets,
        }
    }
}

impl Operator for PackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let mut len = usize::MAX;
        let mut ended = 0;
        for index in 0..self.inlets {
            let mut inlet = cx.inlet(index);
            let closed = inlet.closed();
            match inlet.peek_mut().map(|front| front.len()) {
                Some(front) => len = len.min(front),
                None if closed => ended += 1,
                None => return Ok(Step::Blocked(Blocked::Inlet(index))),
            }
        }
        if ended == self.inlets {
            return Ok(Step::Finished);
        }
        vortex_ensure!(ended == 0, "Pack fields ended at different rows");
        let mut arrays = Vec::with_capacity(self.inlets);
        let mut more = true;
        for index in 0..self.inlets {
            let mut inlet = cx.inlet(index);
            arrays.push(take_rows(&mut inlet, len)?);
            more &= !inlet.is_empty();
        }
        let validity = if self.nullable {
            Validity::Array(
                arrays
                    .pop()
                    .ok_or_else(|| vortex_err!("Nullable Pack is missing its validity"))?,
            )
        } else {
            Validity::NonNullable
        };
        let array = StructArray::try_new_with_dtype(arrays, self.fields.clone(), len, validity)?
            .into_array();
        Ok(if more {
            Step::More(array)
        } else {
            Step::Last(array)
        })
    }
}

impl Source for PackSource {
    fn inlet_count(&self) -> usize {
        self.inlets
    }
}

/// Wraps each batch of a pack's only field into a struct.
pub(crate) struct WrapStage {
    names: FieldNames,
}

impl WrapStage {
    pub(crate) fn new(names: FieldNames) -> Self {
        Self { names }
    }
}

impl Operator for WrapStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => {
                let len = batch.len();
                Ok(Step::Last(
                    StructArray::try_new(
                        self.names.clone(),
                        vec![batch],
                        len,
                        Validity::NonNullable,
                    )?
                    .into_array(),
                ))
            }
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}

/// Applies an expression to each batch.
pub(crate) struct EvalStage {
    expression: BoundExpression,
}

impl EvalStage {
    pub(crate) fn new(expression: BoundExpression) -> Self {
        Self { expression }
    }
}

impl Operator for EvalStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => Ok(Step::Last(batch.apply_bound(&self.expression)?)),
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}

/// Narrows a whole decoded segment, the one batch of a share's port, to a reader's rows.
pub(crate) struct SelectStage {
    slice: Option<Range<usize>>,
    filter: Option<Mask>,
}

impl SelectStage {
    pub(crate) fn new(slice: Option<Range<usize>>, filter: Option<Mask>) -> Self {
        Self { slice, filter }
    }
}

impl Operator for SelectStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => Ok(Step::Last(select(
                batch,
                self.slice.as_ref(),
                self.filter.as_ref(),
            )?)),
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}

/// Keeps the selected rows of the dense batches below it, each by its own slice of the mask.
pub(crate) struct MaskStage {
    mask: Mask,
    cursor: usize,
    predicate: bool,
}

impl MaskStage {
    pub(crate) fn new(mask: Mask, predicate: bool) -> Self {
        Self {
            mask,
            cursor: 0,
            predicate,
        }
    }
}

impl Operator for MaskStage {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => {
                let end = self.cursor + batch.len();
                vortex_ensure!(
                    end <= self.mask.len(),
                    "Filter input is longer than its mask"
                );
                let mask = self.mask.slice(self.cursor..end);
                self.cursor = end;
                Ok(Step::Last(keep_selected(batch, mask, self.predicate, cx)?))
            }
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}

/// Wraps each batch of codes as a dictionary over the values, once the values inlet has ended.
pub(crate) struct TakeSource {
    plan: TakePlan,
    values: Option<ArrayRef>,
}

const CODES: usize = 0;
const VALUES: usize = 1;

impl TakeSource {
    /// A take over `values`, or over what its values inlet produces when `None`.
    pub(crate) fn new(plan: TakePlan, values: Option<ArrayRef>) -> Self {
        Self { plan, values }
    }
}

impl Operator for TakeSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let values = match &self.values {
            Some(values) => values.clone(),
            None => {
                let mut inlet = cx.inlet(VALUES);
                if !inlet.closed() {
                    return Ok(Step::Blocked(Blocked::Inlet(VALUES)));
                }
                let values = join(self.plan.values()?.dtype(), drain(&mut inlet))?;
                let values = if values.is::<Shared>() {
                    values
                } else {
                    SharedArray::new(values).into_array()
                };
                let values = self.plan.cache_values(values);
                self.values = Some(values.clone());
                values
            }
        };
        let mut codes = cx.inlet(CODES);
        match codes.take() {
            Some(batch) => {
                let more = !codes.is_empty();
                let array = DictArray::try_new(batch, values)?.into_array();
                Ok(if more {
                    Step::More(array)
                } else {
                    Step::Last(array)
                })
            }
            None if codes.closed() => Ok(Step::Finished),
            None => Ok(Step::Blocked(Blocked::Inlet(CODES))),
        }
    }
}

impl Source for TakeSource {
    fn inlet_count(&self) -> usize {
        if self.values.is_some() { 1 } else { 2 }
    }

    fn capacity(&self, inlet: usize) -> usize {
        if inlet == VALUES {
            UNBOUNDED
        } else {
            super::DEFAULT_CAPACITY
        }
    }
}

/// Reads list offsets, then the element range they reference, then assembles the lists and
/// keeps the selected ones.
///
/// The element range is known only once the offsets are whole, so the elements are compiled
/// then, into a new inlet.
pub(crate) struct ListPackSource {
    plan: ListPackPlan,
    /// The selection over the lists read.
    mask: Mask,
    has_validity: bool,
    /// The offsets rebased to the element range, once read, and the elements inlet.
    offsets: Option<(ArrayRef, usize)>,
    done: bool,
}

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

impl ListPackSource {
    pub(crate) fn new(plan: ListPackPlan, mask: Mask, has_validity: bool) -> Self {
        Self {
            plan,
            mask,
            has_validity,
            offsets: None,
            done: false,
        }
    }

    fn spawn_elements(&mut self, cx: &mut Cx<'_>) -> VortexResult<()> {
        let offsets_plan = self.plan.offsets()?;
        let offsets = join(offsets_plan.dtype(), drain(&mut cx.inlet(OFFSETS)))?;
        vortex_ensure!(
            offsets.len() == self.mask.len() + 1,
            "Incomplete list offsets"
        );
        let start = offsets
            .execute_scalar(0, cx.exec())?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid first list offset"))?;
        let end = offsets
            .execute_scalar(offsets.len() - 1, cx.exec())?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid last list offset"))?;
        let elements = self.plan.elements()?;
        vortex_ensure!(
            start <= end && end <= elements.row_count(),
            "List element range {start}..{end} is out of bounds"
        );
        let offsets = if start == 0 {
            offsets
        } else {
            let base = vortex_array::arrays::ConstantArray::new(start, offsets.len())
                .into_array()
                .cast(offsets.dtype().clone())?;
            offsets.binary(base, BinaryOperator::Sub)?
        };
        let inlet = cx.spawn(
            elements,
            start..end,
            Mask::new_true(usize::try_from(end - start)?),
        );
        self.offsets = Some((offsets, inlet));
        Ok(())
    }
}

impl Operator for ListPackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::Finished);
        }
        let Some((_, elements)) = &self.offsets else {
            if !cx.inlet(OFFSETS).closed() {
                return Ok(Step::Blocked(Blocked::Inlet(OFFSETS)));
            }
            self.spawn_elements(cx)?;
            return Ok(Step::Consumed);
        };
        let elements = *elements;
        let inlets: &[usize] = if self.has_validity {
            &[VALIDITY, OFFSETS]
        } else {
            &[OFFSETS]
        };
        for &index in inlets.iter().chain([&elements]) {
            if !cx.inlet(index).closed() {
                return Ok(Step::Blocked(Blocked::Inlet(index)));
            }
        }
        let elements = join(
            self.plan.elements()?.dtype(),
            drain(&mut cx.inlet(elements)),
        )?;
        let validity = match self.plan.validity()? {
            Some(plan) => Validity::Array(join(plan.dtype(), drain(&mut cx.inlet(VALIDITY)))?),
            None => Validity::NonNullable,
        };
        let (offsets, _) = self
            .offsets
            .take()
            .ok_or_else(|| vortex_err!("Missing list offsets"))?;
        self.done = true;
        let lists = ListArray::try_new(elements, offsets, validity)?
            .into_array()
            .filter(self.mask.clone())?;
        Ok(Step::Last(lists))
    }
}

impl Source for ListPackSource {
    fn inlet_count(&self) -> usize {
        if self.has_validity { 2 } else { 1 }
    }

    fn capacity(&self, _inlet: usize) -> usize {
        UNBOUNDED
    }
}

/// The struct of no fields over `len` rows.
pub(crate) fn empty_struct(
    fields: StructFields,
    nullability: Nullability,
    len: usize,
) -> VortexResult<ArrayRef> {
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    };
    Ok(StructArray::try_new_with_dtype(Vec::new(), fields, len, validity)?.into_array())
}
