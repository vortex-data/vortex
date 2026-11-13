// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FastLanes;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::filter::filter_chunked;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Delta;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::chunks::DeltaChunks;

/// Filter a Delta array by decoding only the chunks that hold selected values.
///
/// [`filter_chunked`] compacts each decoded chunk while it is in cache.
impl FilterKernel for Delta {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        let validity = array.validity()?.filter(mask)?;
        let bases = array.bases().clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = bases.ptype();
        // Deltas are decoded as unsigned; `wrapping_add` is the same for either signedness.
        let bases = bases.reinterpret_cast(ptype.to_unsigned());
        let filtered = match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
            let mut deltas = None;
            let chunks = DeltaChunks::<T, { T::LANES }>::for_array(
                array,
                bases.as_slice(),
                &mut deltas,
                ctx,
            )?;
            filter_chunked(&chunks, array.offset(), values, ctx.allocator()).into_byte_buffer()
        });
        Ok(Some(
            PrimitiveArray::from_byte_buffer(filtered, ptype, validity).into_array(),
        ))
    }
}
