// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Merge Narrow inputs at a common stored width.
//!
//! Concatenation and indexed interleave retain the logical dtype while consuming narrower values.
//! Interleave selectors can also use their stored integer children because selector width does
//! not determine the result dtype.

use vortex_error::VortexResult;

use super::Narrow;
use super::NarrowArraySlotsExt;
use super::rules::rewrap;
use super::rules::scalar_storage_type;
use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::Interleave;
use crate::arrays::InterleaveArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::chunked::ChunkedSlots;
use crate::arrays::interleave::InterleaveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::optimizer::rules::ArrayParentReduceRule;

#[derive(Debug)]
pub(super) struct ChunkedInputsReduce;

impl ArrayParentReduceRule<Narrow> for ChunkedInputsReduce {
    type Parent = Chunked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Narrow>,
        parent: ArrayView<'_, Chunked>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx < ChunkedSlots::CHUNKS_OFFSET {
            return Ok(None);
        }
        let Some(values) = stored_inputs(array, parent.chunks())? else {
            return Ok(None);
        };
        let Some(first) = values.first() else {
            return Ok(None);
        };
        let dtype = first.dtype().clone();
        rewrap(array, ChunkedArray::try_new(values, dtype)?.into_array())
    }
}

#[derive(Debug)]
pub(super) struct InterleaveInputsReduce;

impl ArrayParentReduceRule<Narrow> for InterleaveInputsReduce {
    type Parent = Interleave;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Narrow>,
        parent: ArrayView<'_, Interleave>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = (0..parent.num_values())
            .map(|idx| parent.value(idx).clone())
            .collect();
        if child_idx < 2 {
            let array_indices = if child_idx == 0 {
                array.values().clone()
            } else {
                parent.array_indices().clone()
            };
            let row_indices = if child_idx == 1 {
                array.values().clone()
            } else {
                parent.row_indices().clone()
            };
            return Ok(Some(
                InterleaveArray::try_new(values, array_indices, row_indices)?.into_array(),
            ));
        }
        let Some(values) = stored_inputs(array, values)? else {
            return Ok(None);
        };
        rewrap(
            array,
            InterleaveArray::try_new(
                values,
                parent.array_indices().clone(),
                parent.row_indices().clone(),
            )?
            .into_array(),
        )
    }
}

/// Casts Narrow and constant inputs to a common stored type, retaining each input's nullability.
fn stored_inputs(
    array: ArrayView<'_, Narrow>,
    inputs: Vec<ArrayRef>,
) -> VortexResult<Option<Vec<ArrayRef>>> {
    let Ok(mut ptype) = PType::try_from(array.values().dtype()) else {
        return Ok(None);
    };
    let mut values = Vec::with_capacity(inputs.len());
    for input in inputs {
        let (stored, width) = if let Some(narrow) = input.as_opt::<Narrow>() {
            let Ok(width) = PType::try_from(narrow.values().dtype()) else {
                return Ok(None);
            };
            (narrow.values().clone(), width)
        } else if let Some(scalar) = input.as_constant() {
            let Some(width) = scalar_storage_type(array, &scalar)? else {
                return Ok(None);
            };
            (input, width)
        } else {
            return Ok(None);
        };
        if width.byte_width() > ptype.byte_width() {
            ptype = width;
        }
        values.push(stored);
    }
    values
        .into_iter()
        .map(|values| values.cast(DType::Primitive(ptype, values.dtype().nullability())))
        .collect::<VortexResult<Vec<_>>>()
        .map(Some)
}
