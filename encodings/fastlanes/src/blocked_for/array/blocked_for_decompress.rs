// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::PrimInt;
use num_traits::WrappingAdd;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::BlockedFoRArray;
use crate::blocked_for::array::BLOCK_SIZE;
use crate::blocked_for::array::BlockedFoRArrayExt;
use crate::blocked_for::array::BlockedFoRArraySlotsExt;

pub fn decompress(array: &BlockedFoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.as_ref().dtype().as_ptype();
    let offset = array.offset() as usize;

    let references = array.references().clone().execute::<PrimitiveArray>(ctx)?;
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = encoded.validity()?;

    Ok(match_each_integer_ptype!(ptype, |T| {
        PrimitiveArray::new(
            decompress_primitive::<T>(
                encoded.into_buffer::<T>(),
                references.as_slice::<T>(),
                offset,
            ),
            validity,
        )
    }))
}

/// Add each block's reference back onto its residuals.
///
/// `offset` is the position of the first value within its block, so the first (possibly short)
/// block covers `BLOCK_SIZE - offset` values and every subsequent block a full `BLOCK_SIZE`.
fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: Buffer<T>,
    references: &[T],
    offset: usize,
) -> Buffer<T> {
    let len = values.len();
    let mut values = values.into_mut();

    let mut pos = 0;
    for (block, reference) in references.iter().enumerate() {
        let block_len = if block == 0 {
            (BLOCK_SIZE - offset).min(len)
        } else {
            BLOCK_SIZE.min(len - pos)
        };
        if !reference.is_zero() {
            for v in &mut values[pos..pos + block_len] {
                *v = v.wrapping_add(reference);
            }
        }
        pos += block_len;
    }
    debug_assert_eq!(pos, len);

    values.freeze()
}
