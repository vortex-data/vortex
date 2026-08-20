// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering an Elias-Fano array through the cursor: the same shape and crossover as
//! [`take`](super::take). Selected rows arrive ascending, which is the cursor's best case.

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::EliasFano;
use crate::EliasFanoCursor;
use crate::compute::take::BULK_DECODE_THRESHOLD;

impl FilterKernel for EliasFano {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let selected = mask
            .values()
            .vortex_expect("FilterKernel precondition: mask is Mask::Values");

        if selected.true_count() * BULK_DECODE_THRESHOLD > array.len() {
            let decoded = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return decoded.into_array().filter(mask.clone()).map(Some);
        }

        let ptype = array.dtype().as_ptype();
        let reference_bits = array.reference_bits();
        let validity = array.validity()?.filter(mask)?;

        let mut cursor = EliasFanoCursor::try_new(array, ctx)?;
        let filtered = match_each_integer_ptype!(ptype, |P| {
            PrimitiveArray::new(
                gather_rows::<P>(&mut cursor, selected.indices(), reference_bits)?,
                validity,
            )
        });
        Ok(Some(filtered.into_array()))
    }
}

/// The selected rows, in the column's own width.
fn gather_rows<P: NativePType>(
    cursor: &mut EliasFanoCursor<'_>,
    positions: &[usize],
    reference_bits: u64,
) -> VortexResult<Buffer<P>>
where
    u64: AsPrimitive<P>,
{
    let mut values = BufferMut::<P>::with_capacity(positions.len());
    for &index in positions {
        let bits = reference_bits.wrapping_add(cursor.access_element(index)?);
        // Truncating the pattern to the column's width is exactly the two's complement result,
        // signed or unsigned, because the reference was added in the same modular arithmetic.
        values.push(bits.as_());
    }
    Ok(values.freeze())
}
