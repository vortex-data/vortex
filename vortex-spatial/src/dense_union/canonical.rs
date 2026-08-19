// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::array::DenseUnion;
use super::array::DenseUnionArrayExt;
use super::array::DenseUnionArraySlotsExt;
use super::tag_lookup;

/// Converts a dense union to its canonical sparse representation.
///
/// A selected variant becomes a dictionary over the original compact child, so values are not
/// copied. Codes for rows that select another variant stay zero, which their type IDs make
/// unreachable. A variant no row selects becomes a constant of its default value.
///
/// # Errors
///
/// Returns an error for unknown type IDs, invalid offsets, or failed array construction.
pub(crate) fn canonicalize(
    array: Array<DenseUnion>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let variants = array.variants().clone();
    let type_ids = array.type_ids().as_::<Primitive>();
    let offsets = array.offsets().as_::<Primitive>();
    let type_id_values = type_ids.as_slice::<u8>();
    let offset_values = offsets.as_slice::<i32>();
    let valid_rows = type_ids.validity()?.execute_mask(len, ctx)?;
    let child_lengths = array.iter_children().map(ArrayRef::len).collect::<Vec<_>>();

    let child_indices = tag_lookup(&variants);
    let mut codes_by_child: Vec<Option<BufferMut<u32>>> = vec![None; variants.len()];

    for (row, ((type_id, offset), valid)) in type_id_values
        .iter()
        .zip(offset_values)
        .zip(valid_rows.iter())
        .enumerate()
    {
        if !valid {
            continue;
        }
        let child_index = child_indices[usize::from(*type_id)]
            .ok_or_else(|| vortex_err!("DenseUnion contains unknown type ID {type_id}"))?;
        let offset = u32::try_from(*offset).map_err(|_| {
            vortex_err!("DenseUnion contains negative offset {offset} at row {row}")
        })?;
        let child_len = child_lengths[child_index];
        vortex_ensure!(
            (offset as usize) < child_len,
            "DenseUnion offset {offset} is out of bounds for child {child_index} of length {child_len}"
        );
        codes_by_child[child_index].get_or_insert_with(|| BufferMut::zeroed(len))[row] = offset;
    }

    let sparse_children = array
        .iter_children()
        .zip(codes_by_child)
        .map(|(child, codes)| {
            // Codes are only recorded for a variant some valid row selects at an in-bounds
            // offset, so a variant without them is unreachable and needs no values at all.
            let Some(codes) = codes else {
                return Ok(
                    ConstantArray::new(Scalar::default_value(child.dtype()), len).into_array(),
                );
            };
            let codes = PrimitiveArray::new(codes.freeze(), Validity::NonNullable).into_array();
            DictArray::try_new(codes, child.clone()).map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    UnionArray::try_new(type_ids.array().clone(), variants, sparse_children)
        .map(IntoArray::into_array)
}
