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
use vortex_mask::AllOr;

use super::array::DenseUnion;
use super::array::DenseUnionArrayExt;
use super::array::DenseUnionArraySlotsExt;

/// Converts a dense union to its canonical sparse representation.
///
/// Each child is a dictionary over the original compact child, so values are not copied. Codes for
/// other variants stay zero because their type IDs make them unreachable. Unused variants use a
/// constant zero code array, and empty children use a one-value constant because dictionaries
/// require non-empty values.
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

    let mut child_indices = [None; 256];
    for (child_index, type_id) in variants.type_ids().iter().copied().enumerate() {
        child_indices[usize::from(type_id)] = Some(child_index);
    }
    let mut codes_by_child: Vec<Option<BufferMut<u32>>> = vec![None; variants.len()];

    let mut assign_row = |row: usize| -> VortexResult<()> {
        let type_id = type_id_values[row];
        let child_index = child_indices[usize::from(type_id)]
            .ok_or_else(|| vortex_err!("DenseUnion contains unknown type ID {type_id}"))?;
        let offset = u32::try_from(offset_values[row]).map_err(|_| {
            vortex_err!(
                "DenseUnion contains negative offset {} at row {row}",
                offset_values[row]
            )
        })?;
        let child_len = child_lengths[child_index];
        vortex_ensure!(
            (offset as usize) < child_len,
            "DenseUnion offset {offset} is out of bounds for child {child_index} of length {child_len}"
        );
        let codes = codes_by_child[child_index].get_or_insert_with(|| BufferMut::zeroed(len));
        codes[row] = offset;
        Ok(())
    };

    match valid_rows.indices() {
        AllOr::All => {
            for row in 0..len {
                assign_row(row)?;
            }
        }
        AllOr::None => {}
        AllOr::Some(rows) => {
            for &row in rows {
                assign_row(row)?;
            }
        }
    }

    let sparse_children = array
        .iter_children()
        .zip(codes_by_child)
        .map(|(child, codes)| {
            let codes = match codes {
                Some(codes) => {
                    PrimitiveArray::new(codes.freeze(), Validity::NonNullable).into_array()
                }
                None => ConstantArray::new(0u32, len).into_array(),
            };
            let values = if child.is_empty() {
                ConstantArray::new(Scalar::default_value(child.dtype()), 1).into_array()
            } else {
                child.clone()
            };
            DictArray::try_new(codes, values).map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    UnionArray::try_new(type_ids.array().clone(), variants, sparse_children)
        .map(IntoArray::into_array)
}
