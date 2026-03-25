// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FoR;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::FoRArray;
use crate::bitpack_decompress;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::UnpackedChunks;

/// FoR unpacking strategy that applies a reference value during unpacking.
struct FoRStrategy<T> {
    reference: T,
}

impl<T: PhysicalPType<Physical = T> + FoR> UnpackStrategy<T> for FoRStrategy<T> {
    #[inline(always)]
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: Caller ensures chunk and dst have correct sizes.
        unsafe {
            FoR::unchecked_unfor_pack(bit_width, chunk, self.reference, dst);
        }
    }
}

pub fn decompress(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();

    // Try to do fused unpack.
    if array.reference_scalar().dtype().is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPacked>()
    {
        return match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            fused_decompress::<T>(array, bp, ctx)
        });
    }

    // TODO(ngates): Do we need this to be into_encoded() somehow?
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = encoded.validity().clone();

    Ok(match_each_integer_ptype!(ptype, |T| {
        let min = array
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("reference must be non-null");
        if min == 0 {
            encoded
        } else {
            PrimitiveArray::new(
                decompress_primitive(encoded.into_buffer::<T>(), min),
                validity,
            )
        }
    }))
}

pub(crate) fn fused_decompress<
    T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd,
>(
    for_: &FoRArray,
    bp: &BitPackedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ref_ = for_
        .reference_scalar()
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let strategy = FoRStrategy { reference: ref_ };

    // Create [`UnpackedChunks`] with FoR strategy.
    let mut unpacked = UnpackedChunks::new_with_strategy(
        strategy,
        bp.packed().as_host().clone(),
        bp.bit_width() as usize,
        bp.offset() as usize,
        bp.len(),
    );

    let mut builder = PrimitiveBuilder::<T>::with_capacity(
        for_.reference_scalar().dtype().nullability(),
        bp.len(),
    );
    let mut uninit_range = builder.uninit_range(bp.len());
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(bp.validity_mask()?);
    }

    // SAFETY: `decode_into` will initialize all values in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, bp.len()) };

    // Decode all chunks (initial, full, and trailer) in one call.
    unpacked.decode_into(uninit_slice);

    if let Some(patches) = bp.patches() {
        bitpack_decompress::apply_patches_to_uninit_range_fn(
            &mut uninit_range,
            patches,
            ctx,
            |v| v.wrapping_add(&ref_),
        )?;
    };

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via `decode_into`.
    unsafe {
        uninit_range.finish();
    }

    Ok(builder.finish_into_primitive())
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: Buffer<T>,
    min: T,
) -> Buffer<T> {
    values
        .map_each_in_place(move |v| v.wrapping_add(&min))
        .freeze()
}
