// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;

use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::chunk_range;
use vortex_array::arrays::primitive::patch_chunk;
use vortex_array::dtype::DType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesParts;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPArrayOwnedExt;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// Decompresses an ALP-encoded array using `to_primitive` (legacy path).
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn decompress_into_array(
    array: ALPArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let dtype = array.dtype().clone();
    let (encoded, exponents, patches) = ALPArrayOwnedExt::into_parts(array);
    let encoded = encoded.execute::<PrimitiveArray>(ctx)?;

    match patches {
        Some(patches) if patches.chunk_offsets().is_some() => {
            let PatchesParts {
                offset,
                indices,
                values,
                chunk_offsets,
                offset_within_chunk,
                ..
            } = patches.into_parts();
            let chunk_offsets = chunk_offsets.vortex_expect("chunk offsets checked above");

            let patches = ChunkedPatches {
                indices: &indices.execute::<PrimitiveArray>(ctx)?,
                values: &values.execute::<PrimitiveArray>(ctx)?,
                chunk_offsets: &chunk_offsets.execute::<PrimitiveArray>(ctx)?,
                offset,
                offset_within_chunk: offset_within_chunk.unwrap_or(0),
            };

            Ok(decompress_chunked_core(encoded, exponents, patches, dtype))
        }
        patches => decompress_unchunked_core(encoded, exponents, patches, dtype, ctx),
    }
}

/// Decompresses an ALP-encoded array on the execution path.
///
/// The encoded child and the patch children must already be primitive arrays, as
/// `ALP::execute` requires.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn execute_decompress(array: ALPArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let dtype = array.dtype().clone();
    let (encoded, exponents, patches) = ALPArrayOwnedExt::into_parts(array);
    let encoded = encoded.downcast::<Primitive>();

    match patches {
        Some(patches) if patches.chunk_offsets().is_some() => {
            let PatchesParts {
                offset,
                indices,
                values,
                chunk_offsets,
                offset_within_chunk,
                ..
            } = patches.into_parts();
            let chunk_offsets = chunk_offsets.vortex_expect("chunk offsets checked above");

            let patches = ChunkedPatches {
                indices: &indices.downcast::<Primitive>(),
                values: &values.downcast::<Primitive>(),
                chunk_offsets: &chunk_offsets.downcast::<Primitive>(),
                offset,
                offset_within_chunk: offset_within_chunk.unwrap_or(0),
            };

            Ok(decompress_chunked_core(encoded, exponents, patches, dtype))
        }
        patches => decompress_unchunked_core(encoded, exponents, patches, dtype, ctx),
    }
}

/// Resolved patch children of a chunked ALP array.
struct ChunkedPatches<'a> {
    indices: &'a PrimitiveArray,
    values: &'a PrimitiveArray,
    chunk_offsets: &'a PrimitiveArray,
    offset: usize,
    offset_within_chunk: usize,
}

/// Core decompression logic for chunked ALP arrays.
///
/// Takes pre-resolved `PrimitiveArray` inputs to avoid duplication between
/// the `to_primitive` and `execute` paths.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from nested match_each_* macros"
)]
fn decompress_chunked_core(
    encoded: PrimitiveArray,
    exponents: Exponents,
    patches: ChunkedPatches<'_>,
    dtype: DType,
) -> PrimitiveArray {
    let ChunkedPatches {
        indices: patches_indices,
        values: patches_values,
        chunk_offsets: patches_chunk_offsets,
        offset: patches_offset,
        offset_within_chunk,
    } = patches;

    let validity = encoded
        .validity()
        .vortex_expect("ALP validity should be derivable");
    let ptype = dtype.as_ptype();
    let array_len = encoded.len();

    match_each_alp_float_ptype!(ptype, |T| {
        let patches_values = patches_values.as_slice::<T>();
        let mut alp_buffer = encoded.into_buffer_mut();
        match_each_unsigned_integer_ptype!(patches_chunk_offsets.ptype(), |C| {
            let patches_chunk_offsets = patches_chunk_offsets.as_slice::<C>();

            match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
                let patches_indices = patches_indices.as_slice::<I>();

                for chunk_idx in 0..patches_chunk_offsets.len() {
                    let chunk_range = chunk_range(chunk_idx, patches_offset, array_len);
                    let chunk_slice = &mut alp_buffer.as_mut_slice()[chunk_range];

                    <T>::decode_slice_inplace(chunk_slice, exponents);

                    let decoded_chunk: &mut [T] = unsafe { transmute(chunk_slice) };
                    patch_chunk(
                        decoded_chunk,
                        patches_indices,
                        patches_values,
                        patches_offset,
                        patches_chunk_offsets,
                        chunk_idx,
                        offset_within_chunk,
                    );
                }

                let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
                PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
            })
        })
    })
}

/// Core decompression logic for unchunked ALP arrays.
///
/// Takes a pre-resolved `PrimitiveArray` to avoid duplication between
/// the `to_primitive` and `execute` paths.
fn decompress_unchunked_core(
    encoded: PrimitiveArray,
    exponents: Exponents,
    patches: Option<Patches>,
    dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = encoded.validity()?;
    let ptype = dtype.as_ptype();

    let decoded = match_each_alp_float_ptype!(ptype, |T| {
        let mut alp_buffer = encoded.into_buffer_mut();
        <T>::decode_slice_inplace(alp_buffer.as_mut_slice(), exponents);
        let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
        PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
    });

    if let Some(patches) = patches {
        decoded.patch(&patches, ctx)
    } else {
        Ok(decoded)
    }
}
