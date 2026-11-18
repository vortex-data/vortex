// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd_macros::simd;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::chunk_range;
use vortex_array::arrays::primitive::patch_chunk;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ALPArray;
use crate::ALPArrayOwnedExt;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// Decompresses an ALP-encoded array whose children are already `PrimitiveArray`s.
///
/// The function takes ownership of the array, so the encoded buffer is decoded in place when no
/// other reference to it exists.
///
/// # Errors
///
/// Returns an error if the encoded child or a patch child is not a `PrimitiveArray`.
pub fn execute_decompress(array: ALPArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype().as_ptype();
    let (encoded, exponents, patches) = ALPArrayOwnedExt::into_parts(array);
    let encoded = encoded
        .try_downcast::<Primitive>()
        .map_err(|array| not_primitive(&array, "encoded"))?;

    let Some(patches) = patches else {
        return decode(encoded, ptype, exponents);
    };

    if patches.chunk_offsets().is_some() {
        decompress_chunked(encoded, ptype, exponents, &patches)
    } else {
        decode(encoded, ptype, exponents)?.patch(&patches, ctx)
    }
}

fn not_primitive(array: &ArrayRef, name: &str) -> VortexError {
    vortex_err!(
        "ALP {name} child must be executed to Primitive before decompression, got {}",
        array.encoding_id()
    )
}

/// Decodes all values of the encoded array in place.
fn decode(
    encoded: PrimitiveArray,
    ptype: PType,
    exponents: Exponents,
) -> VortexResult<PrimitiveArray> {
    let validity = encoded.validity()?;

    Ok(match_each_alp_float_ptype!(ptype, |T| {
        let mut alp_buffer = encoded.into_buffer_mut();
        decode_inplace::<T>(Level::new(), alp_buffer.as_mut_slice(), exponents);

        // SAFETY: `T::ALPInt` has the same size and alignment as `T`, and every bit pattern is a
        // valid `T`.
        let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
        PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
    }))
}

/// Decodes the encoded array one patch chunk at a time and applies the patches of each chunk
/// while the chunk is still in cache.
fn decompress_chunked(
    encoded: PrimitiveArray,
    ptype: PType,
    exponents: Exponents,
    patches: &Patches,
) -> VortexResult<PrimitiveArray> {
    let validity = encoded.validity()?;

    let patches_chunk_offsets = patches
        .chunk_offsets()
        .as_ref()
        .ok_or_else(|| vortex_err!("ALP chunked decompression requires patch chunk offsets"))?;

    let patches_chunk_offsets = patches_chunk_offsets
        .as_opt::<Primitive>()
        .ok_or_else(|| not_primitive(patches_chunk_offsets, "patch chunk offsets"))?;
    let patches_indices = patches
        .indices()
        .as_opt::<Primitive>()
        .ok_or_else(|| not_primitive(patches.indices(), "patch indices"))?;
    let patches_values = patches
        .values()
        .as_opt::<Primitive>()
        .ok_or_else(|| not_primitive(patches.values(), "patch values"))?;

    Ok(match_each_alp_float_ptype!(ptype, |T| {
        let mut alp_buffer = encoded.into_buffer_mut();

        match_each_unsigned_integer_ptype!(patches_chunk_offsets.ptype(), |C| {
            match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
                decode_and_patch_chunks::<T, I, C>(
                    alp_buffer.as_mut_slice(),
                    exponents,
                    patches,
                    patches_indices.as_slice::<I>(),
                    patches_values.as_slice::<T>(),
                    patches_chunk_offsets.as_slice::<C>(),
                )
            })
        });

        // SAFETY: `T::ALPInt` has the same size and alignment as `T`, and every bit pattern is a
        // valid `T`.
        let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
        PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
    }))
}

/// Decodes `values` in place one patch chunk at a time, applying each chunk's patches.
fn decode_and_patch_chunks<T, I, C>(
    values: &mut [T::ALPInt],
    exponents: Exponents,
    patches: &Patches,
    patches_indices: &[I],
    patches_values: &[T],
    patches_chunk_offsets: &[C],
) where
    T: ALPFloat + NativePType,
    I: UnsignedPType,
    C: UnsignedPType,
{
    let array_len = values.len();
    let offset_within_chunk = patches.offset_within_chunk().unwrap_or(0);
    let level = Level::new();

    for chunk_idx in 0..patches_chunk_offsets.len() {
        let chunk_range = chunk_range(chunk_idx, patches.offset(), array_len);
        let chunk_slice = &mut values[chunk_range];

        decode_inplace::<T>(level, chunk_slice, exponents);

        // SAFETY: `ALPFloat` is sealed to `f32` and `f64`, whose `ALPInt` (`i32` or `i64`) has the
        // same size and alignment, and both types accept every bit pattern.
        let decoded_chunk: &mut [T] = unsafe { transmute(chunk_slice) };
        patch_chunk(
            decoded_chunk,
            patches_indices,
            patches_values,
            patches.offset(),
            patches_chunk_offsets,
            chunk_idx,
            offset_within_chunk,
        );
    }
}

/// Decodes `encoded` in place, compiled for the SIMD `level`.
#[inline]
fn decode_inplace<T: ALPFloat>(level: Level, encoded: &mut [T::ALPInt], exponents: Exponents) {
    dispatch!(level, simd => decode_inplace_simd::<_, T>(simd, encoded, exponents))
}

/// The `alp` decode loop is inlined into this body, so each level gets its own instantiation that
/// LLVM vectorizes with that level's target features rather than the compilation baseline.
#[simd]
fn decode_inplace_simd<S: Simd, T: ALPFloat>(
    _: S,
    encoded: &mut [T::ALPInt],
    exponents: Exponents,
) {
    T::decode_slice_inplace(encoded, exponents);
}
