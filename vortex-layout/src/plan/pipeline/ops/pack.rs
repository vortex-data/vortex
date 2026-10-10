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

use super::*;
use crate::plan::pipeline::Blocked;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;

/// Zips its inlets, one per field and the validity last when nullable, into structs.
///
/// When every inlet's front batch has the same length, the fields are chunked alike there and the
/// struct takes those batches whole. Otherwise the struct takes as many rows as every inlet has
/// queued, once the inlet with the fewest can queue no more, so fields chunked differently are
/// joined, not cut at every boundary of every field.
///
/// Waiting costs nothing per wake. Queued rows only grow until the struct is built, so the source
/// resumes at the first inlet it found empty instead of rescanning the others, and then waits on
/// the one inlet with the fewest rows.
pub(crate) struct PackSource {
    fields: StructFields,
    nullable: bool,
    inlets: usize,
    /// Inlets before this one hold rows.
    cursor: usize,
    /// The fewest rows queued on an inlet before `cursor` when the scan saw it, and that inlet.
    fewest: (usize, usize),
    /// The length of the front batch of every inlet before `cursor`, while they agree.
    front: Option<usize>,
    /// Inlets before `cursor` that had ended.
    ended: usize,
    /// Every inlet holds rows, and the source waits for this one, which holds the fewest, to
    /// fill or close.
    filling: Option<usize>,
}

impl PackSource {
    pub(crate) fn new(fields: StructFields, nullable: bool, inlets: usize) -> Self {
        Self {
            fields,
            nullable,
            inlets,
            cursor: 0,
            fewest: (usize::MAX, 0),
            front: None,
            ended: 0,
            filling: None,
        }
    }

    /// Finds whether every inlet holds rows, resuming at `cursor`.
    fn scan(&mut self, cx: &mut Cx<'_>) -> Option<Step> {
        while self.cursor < self.inlets {
            let mut inlet = cx.inlet(self.cursor);
            match inlet.peek_mut().map(|front| front.len()) {
                None if inlet.closed() => self.ended += 1,
                None => return Some(Step::Blocked(Blocked::Inlet(self.cursor))),
                Some(front) => {
                    self.front = match self.cursor {
                        0 => Some(front),
                        _ => self.front.filter(|&agreed| agreed == front),
                    };
                    let rows = inlet.rows();
                    if rows < self.fewest.0 {
                        self.fewest = (rows, self.cursor);
                    }
                }
            }
            self.cursor += 1;
        }
        None
    }
}

impl Operator for PackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if let Some(index) = self.filling {
            let inlet = cx.inlet(index);
            if !inlet.full() && !inlet.closed() {
                return Ok(Step::Blocked(Blocked::Inlet(index)));
            }
            // Every inlet may have gained rows while the source waited.
            self.filling = None;
            self.cursor = 0;
            self.fewest = (usize::MAX, 0);
            self.front = None;
        }
        if let Some(blocked) = self.scan(cx) {
            return Ok(blocked);
        }
        if self.ended == self.inlets {
            return Ok(Step::Finished);
        }
        vortex_ensure!(self.ended == 0, "Pack fields ended at different rows");
        let fewest = self.fewest.1;
        let len = match self.front {
            Some(front) => front,
            None => {
                let inlet = cx.inlet(fewest);
                if !inlet.full() && !inlet.closed() {
                    self.filling = Some(fewest);
                    return Ok(Step::Blocked(Blocked::Inlet(fewest)));
                }
                // The counts the scan saw are lower bounds by now, so count again.
                (0..self.inlets)
                    .map(|index| cx.inlet(index).rows())
                    .min()
                    .unwrap_or(0)
            }
        };
        self.cursor = 0;
        self.fewest = (usize::MAX, 0);
        self.front = None;
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
        let array = assemble(&self.fields, arrays, validity).into_array();
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
    fields: StructFields,
}

impl WrapStage {
    pub(crate) fn new(fields: StructFields) -> Self {
        Self { fields }
    }
}

impl Operator for WrapStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => Ok(Step::Last(
                assemble(&self.fields, vec![batch], Validity::NonNullable).into_array(),
            )),
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
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
