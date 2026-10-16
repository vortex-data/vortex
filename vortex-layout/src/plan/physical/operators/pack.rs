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
use vortex_error::vortex_err;

use super::take_rows;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::InletSet;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
use crate::plan::physical::pipeline::Inlet;

#[derive(Clone, Copy)]
enum InletState {
    Waiting,
    Ready,
    Ended,
}

/// Zips its inlets, one per field and the validity last when nullable, into structs.
///
/// Readiness updates only the inlets whose sinks changed them. Computing takes the shortest
/// front batch's length from every inlet, retaining the remainder of longer batches.
pub(crate) struct PackSource {
    fields: StructFields,
    nullable: bool,
    states: Vec<InletState>,
    waiting: InletSet,
    ended: usize,
}

impl PackSource {
    pub(crate) fn new(fields: StructFields, nullable: bool, inlets: usize) -> Self {
        Self {
            fields,
            nullable,
            states: vec![InletState::Waiting; inlets],
            waiting: InletSet::all(inlets),
            ended: 0,
        }
    }

    fn observe(&mut self, index: usize, inlet: &Inlet<'_>) {
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
}

impl Source for PackSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        while let Some(index) = cx.changed_inlet() {
            self.observe(index, &cx.inlet(index));
        }
        let available = self.states.len() - self.waiting.indices().len() - self.ended;
        vortex_ensure!(
            self.ended == 0 || available == 0,
            AssertionFailed: "Pack fields ended at different rows"
        );
        Ok(match self.waiting.indices().last() {
            Some(&index) => SourceReady::Blocked(SourceBlocked::Inlet(index)),
            None => SourceReady::Ready,
        })
    }
}

impl Operator for PackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.ended == self.states.len() {
            return Ok(Step::End);
        }
        // TODO: Revisit whether waiting to coalesce misaligned batches is worth producing
        // fewer, larger structs instead of emitting at every field's batch boundary.
        let mut len = usize::MAX;
        for index in 0..self.states.len() {
            let mut inlet = cx.inlet(index);
            let front = inlet.peek_mut().ok_or_else(
                || vortex_err!(AssertionFailed: "Pack computed before every inlet was ready"),
            )?;
            len = len.min(front.len());
        }
        let mut arrays = Vec::with_capacity(self.states.len());
        for index in 0..self.states.len() {
            let mut inlet = cx.inlet(index);
            arrays.push(take_rows(&mut inlet, len)?);
            self.observe(index, &inlet);
        }
        let validity = if self.nullable {
            Validity::Array(arrays.pop().ok_or_else(
                || vortex_err!(AssertionFailed: "Nullable Pack is missing its validity"),
            )?)
        } else {
            Validity::NonNullable
        };
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
