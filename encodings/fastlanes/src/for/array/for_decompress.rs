// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::mem;
use std::mem::MaybeUninit;

use fastlanes::FoR;
use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::FoRArray;
use crate::bitpack_decompress;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::UnpackedChunks;
use crate::unpack_iter::for_each_packed_chunk;

/// FoR unpacking strategy that applies a reference value during unpacking.
struct FoRStrategy<T> {
    reference: T,
}

impl<T: PhysicalPType<Physical = T> + FoR> UnpackStrategy<T> for FoRStrategy<T> {
    #[allow(clippy::inline_always)]
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
    match array.constant_reference() {
        Some(reference) => decompress_one_ref(array, &reference, ctx),
        None => decompress_many_refs(array, ctx),
    }
}

/// Decompress an array whose chunks all share `reference`.
fn decompress_one_ref(
    array: &FoRArray,
    reference: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();

    // Try to do fused unpack.
    if ptype.is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPacked>()
    {
        return match_each_unsigned_integer_ptype!(ptype, |T| {
            fused_decompress::<T>(array, bp, ctx)
        });
    }

    // TODO(ngates): Do we need this to be into_encoded() somehow?
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = encoded.validity()?;

    Ok(match_each_integer_ptype!(ptype, |T| {
        let min = reference
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

/// Decompress an array whose chunks have different references.
fn decompress_many_refs(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();

    // Try to do fused unpack. BitPacked chunks line up with FoR chunks when the offsets match.
    if ptype.is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPacked>()
        && bp.offset() == array.offset()
    {
        return match_each_unsigned_integer_ptype!(ptype, |T| {
            fused_decompress_many_refs::<T>(array, bp, ctx)
        });
    }

    match_each_integer_ptype!(ptype, |T| { add_references::<T>(array, ctx) })
}

/// Decode `encoded`, then add each chunk's reference in place.
fn add_references<T: NativePType + WrappingAdd + PrimInt>(
    array: &FoRArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    if encoded.is_empty() {
        return Ok(encoded);
    }
    let validity = encoded.validity()?;
    let references = array.references().clone().execute::<PrimitiveArray>(ctx)?;
    let references = references.as_slice::<T>();

    // The first chunk may be partial when the array was sliced.
    let first_len = (FL_CHUNK_SIZE - usize::from(array.offset())).min(array.len());
    let mut values = encoded.into_buffer_mut::<T>();
    let (first, rest) = values.as_mut_slice().split_at_mut(first_len);
    let chunks = iter::once(first).chain(rest.chunks_mut(FL_CHUNK_SIZE));
    for (chunk, &reference) in chunks.zip_eq(references) {
        for value in chunk {
            *value = value.wrapping_add(&reference);
        }
    }
    Ok(PrimitiveArray::new(values.freeze(), validity))
}

pub(crate) fn fused_decompress<
    T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ref_ = for_
        .constant_reference()
        .ok_or_else(|| vortex_err!("fused FoR decompression requires a constant reference"))?
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let strategy = FoRStrategy { reference: ref_ };
    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];

    // Create [`UnpackedChunks`] with FoR strategy.
    let mut unpacked = UnpackedChunks::try_new_with_strategy(
        strategy,
        bp.packed_slice::<T>(),
        bp.bit_width() as usize,
        bp.offset() as usize,
        bp.len(),
        &mut scratch,
    )?;

    let mut builder = PrimitiveBuilder::<T>::with_capacity_in(
        for_.dtype().nullability(),
        bp.len(),
        ctx.allocator(),
    );
    let mut uninit_range = builder.uninit_range(bp.len());
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(&bp.validity()?.execute_mask(bp.as_ref().len(), ctx)?);
    }

    // SAFETY: `decode_into` will initialize all values in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, bp.len()) };

    // Decode all chunks (initial, full, and trailer) in one call.
    unpacked.decode_into(uninit_slice);

    if let Some(patches) = bp.patches() {
        bitpack_decompress::apply_patches_to_uninit_range(
            &mut uninit_range,
            &patches,
            ctx,
            |v: T| v.wrapping_add(&ref_),
        )?;
    };

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via `decode_into`.
    unsafe {
        uninit_range.finish();
    }

    Ok(builder.finish_into_primitive())
}

/// Unpack each BitPacked chunk and add its chunk's reference in one pass.
fn fused_decompress_many_refs<
    T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let references = for_.references().clone().execute::<PrimitiveArray>(ctx)?;
    let references = references.as_slice::<T>();
    let offset = usize::from(for_.offset());
    let bit_width = bp.bit_width() as usize;
    let len = bp.len();

    let mut builder =
        PrimitiveBuilder::<T>::with_capacity_in(for_.dtype().nullability(), len, ctx.allocator());
    let mut uninit_range = builder.uninit_range(len);
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(&bp.validity()?.execute_mask(len, ctx)?);
    }

    // SAFETY: the loop below initializes every value in this range.
    let output = unsafe { uninit_range.slice_uninit_mut(0, len) };
    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    for_each_packed_chunk::<T, _>(
        bp.packed_slice::<T>(),
        bit_width,
        offset,
        len,
        |packed, range| {
            let reference = references[range.start / FL_CHUNK_SIZE];
            // `range` counts from the start of the first chunk, and the output starts at `offset`.
            let skip = offset.saturating_sub(range.start);
            let dst = &mut output[range.start + skip - offset..range.end - offset];
            if dst.len() == FL_CHUNK_SIZE {
                // SAFETY: `packed` holds one chunk at `bit_width` and `dst` has room for a chunk.
                unsafe {
                    FoR::unchecked_unfor_pack(
                        bit_width,
                        packed,
                        reference,
                        mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(dst),
                    );
                }
            } else {
                // SAFETY: as above, with `scratch` as the destination.
                unsafe {
                    FoR::unchecked_unfor_pack(
                        bit_width,
                        packed,
                        reference,
                        mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(&mut scratch[..]),
                    );
                }
                dst.copy_from_slice(&scratch[skip..range.len()]);
            }
        },
    )?;

    if let Some(patches) = bp.patches() {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let values = values.as_slice::<T>();
        match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
            for (&index, &value) in indices.as_slice::<P>().iter().zip_eq(values) {
                let index = <P as AsPrimitive<usize>>::as_(index) - patches.offset();
                let reference = references[(offset + index) / FL_CHUNK_SIZE];
                uninit_range.set_value(index, value.wrapping_add(&reference));
            }
        });
    }

    // SAFETY: We have set a correct validity mask via `append_mask` with `len` values and
    // initialized every value in the loop above.
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
