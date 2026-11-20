// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zipping fields into structs.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::execution::operator::Cx;
use crate::execution::operator::InletSet;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::take_rows;
use crate::execution::pipeline::Inlet;

#[derive(Clone, Copy)]
enum InletState {
    Waiting,
    Ready,
    Ended,
}

/// Zips its inlets, the validity first when nullable and then one per field, into structs.
///
/// The inlets are the present slots of a [`Pack`](crate::plan::Pack) plan in slot order.
///
/// Readiness updates only the inlets whose sinks changed them. Fields whose batches end at
/// different rows are coalesced: the pack waits until every inlet that can still take a batch
/// holds the rows of the longest front batch, then takes the same rows from every inlet, so a
/// field is cut once per struct rather than at every other field's batch boundary.
pub(crate) struct PackSource {
    fields: StructFields,
    nullable: bool,
    states: Vec<InletState>,
    queued: Vec<Queued>,
    waiting: InletSet,
    ended: usize,
}

/// What an inlet held when it was last observed.
#[derive(Clone, Copy, Default)]
struct Queued {
    front: usize,
    rows: usize,
    /// Its sink can still push without the pack taking a batch first.
    room: bool,
}

impl PackSource {
    pub(crate) fn new(fields: StructFields, nullable: bool, inlets: usize) -> Self {
        Self {
            fields,
            nullable,
            states: vec![InletState::Waiting; inlets],
            queued: vec![Queued::default(); inlets],
            waiting: InletSet::all(inlets),
            ended: 0,
        }
    }

    fn observe(&mut self, index: usize, inlet: &mut Inlet<'_>) {
        self.queued[index] = Queued {
            front: inlet.peek_mut().map_or(0, |front| front.len()),
            rows: inlet.rows(),
            room: !inlet.closed() && !inlet.full(),
        };
        self.ended -= usize::from(matches!(self.states[index], InletState::Ended));
        self.states[index] = if !inlet.is_empty() {
            self.waiting.remove(index);
            InletState::Ready
        } else if inlet.closed() {
            self.waiting.remove(index);
            self.ended += 1;
            InletState::Ended
        } else {
            self.waiting.insert(index);
            InletState::Waiting
        };
    }

    fn longest_front(&self) -> usize {
        self.queued
            .iter()
            .map(|queued| queued.front)
            .max()
            .unwrap_or(0)
    }
}

impl Source for PackSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        while let Some(index) = cx.changed_inlet() {
            self.observe(index, &mut cx.inlet(index));
        }
        let available = self.states.len() - self.waiting.indices().len() - self.ended;
        vortex_ensure!(
            self.ended == 0 || available == 0,
            "Pack fields ended at different rows"
        );
        if let Some(&index) = self.waiting.indices().last() {
            return Ok(SourceReady::Blocked(SourceBlocked::Inlet(index)));
        }
        let longest = self.longest_front();
        Ok(
            match self
                .queued
                .iter()
                .position(|queued| queued.room && queued.rows < longest)
            {
                Some(index) => SourceReady::Blocked(SourceBlocked::Inlet(index)),
                None => SourceReady::Ready,
            },
        )
    }
}

impl Operator for PackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.ended == self.states.len() {
            return Ok(Step::End);
        }
        vortex_ensure!(
            self.waiting.indices().is_empty() && self.ended == 0,
            "Pack computed before every inlet was ready"
        );
        let len = self
            .queued
            .iter()
            .map(|queued| queued.rows)
            .fold(self.longest_front(), usize::min);
        let session = cx.session().clone();
        let inlets = self.states.len();
        let nullable = self.nullable;
        let mut take = |index: usize| -> VortexResult<ArrayRef> {
            let mut inlet = cx.inlet(index);
            let array = take_rows(&mut inlet, len, &session)?;
            self.observe(index, &mut inlet);
            Ok(array)
        };
        let validity = if nullable {
            Validity::Array(take(0)?)
        } else {
            Validity::NonNullable
        };
        let arrays = (usize::from(nullable)..inlets)
            .map(&mut take)
            .collect::<VortexResult<Vec<_>>>()?;
        let array = assemble(&self.fields, arrays, validity).into_array();
        Ok(Step::Final(array))
    }
}

/// Wraps each batch of a pack's only field into a struct.
pub(crate) struct WrapStage {
    fields: StructFields,
}

impl WrapStage {
    pub(crate) fn new(fields: StructFields) -> Self {
        Self { fields }
    }
}

impl Operator for WrapStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        Ok(Step::Final(
            assemble(&self.fields, vec![batch], Validity::NonNullable).into_array(),
        ))
    }
}

/// A struct of `fields` over arrays of one length.
///
/// The type is not derived or validated per batch: a `Pack` plan checks when it is built that
/// each field's plan produces the field's dtype over the struct's rows, and a pack source takes
/// the same rows from every field.
fn assemble(fields: &StructFields, arrays: Vec<ArrayRef>, validity: Validity) -> StructArray {
    let len = arrays.first().map_or(0, |array| array.len());
    debug_assert!(
        arrays.iter().all(|array| array.len() == len)
            && fields
                .fields()
                .zip(&arrays)
                .all(|(dtype, array)| &dtype == array.dtype()),
        "pack fields do not match the struct"
    );
    // SAFETY: the plan validated every field's dtype against `fields`, and every array holds the
    // same rows, checked above in debug builds.
    unsafe { StructArray::new_unchecked(arrays, fields.clone(), len, validity) }
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
