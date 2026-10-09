// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zipping fields into structs.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
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
