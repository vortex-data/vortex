// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;

use vortex_array::ArrayRef;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::chunk_range;
use vortex_array::arrays::patch_chunk;
use vortex_array::patches::Patches;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::match_each_unsigned_integer_ptype;

use crate::ALPArray;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// Decompresses an ALP-encoded array.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn decompress_into_array(array: ALPArray) -> PrimitiveArray {
    let (encoded, exponents, patches, dtype) = array.into_parts();
    if let Some(ref patches) = patches
        && let Some(chunk_offsets) = patches.chunk_offsets()
    {
        decompress_chunked(
            encoded,
            exponents,
            patches,
            &chunk_offsets.as_ref().to_primitive(),
            dtype,
        )
    } else {
        decompress_unchunked(encoded, exponents, patches, dtype)
    }
}

/// Decompresses an ALP-encoded array in 1024-element chunks.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed values with all patches applied.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from nested match_each_* macros"
)]
fn decompress_chunked(
    array: ArrayRef,
    exponents: Exponents,
    patches: &Patches,
    patches_chunk_offsets: &PrimitiveArray,
    dtype: DType,
) -> PrimitiveArray {
    let encoded = array.to_primitive();

    let validity = encoded.validity().clone();

    let patches_indices = patches.indices().to_primitive();
    let patches_values = patches.values().to_primitive();
    let ptype = dtype.as_ptype();
    let array_len = array.len();

    // Number of patches to skip at the start of the first chunk.
    let offset_within_chunk = patches.offset_within_chunk().unwrap_or(0);

    // We need to drop ALPArray here in case converting encoded buffer into
    // primitive didn't create a copy. In that case both alp_encoded and array
    // will hold a reference to the buffer we want to mutate.
    drop(array);

    match_each_alp_float_ptype!(ptype, |T| {
        let patches_values = patches_values.as_slice::<T>();
        let mut alp_buffer = encoded.into_buffer_mut();
        match_each_unsigned_integer_ptype!(patches_chunk_offsets.ptype(), |C| {
            let patches_chunk_offsets = patches_chunk_offsets.as_slice::<C>();

            match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
                let patches_indices = patches_indices.as_slice::<I>();

                for chunk_idx in 0..patches_chunk_offsets.len() {
                    let chunk_range = chunk_range(chunk_idx, patches.offset(), array_len);
                    let chunk_slice = &mut alp_buffer.as_mut_slice()[chunk_range];

                    <T>::decode_slice_inplace(chunk_slice, exponents);

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

                let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
                PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
            })
        })
    })
}

/// Decompresses an ALP-encoded array without chunk offsets.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed values with all patches applied.
fn decompress_unchunked(
    array: ArrayRef,
    exponents: Exponents,
    patches: Option<Patches>,
    dtype: DType,
) -> PrimitiveArray {
    let encoded = array.to_primitive();

    // We need to drop ALPArray here in case converting encoded buffer into
    // primitive didn't create a copy. In that case both alp_encoded and array
    // will hold a reference to the buffer we want to mutate.
    drop(array);

    let validity = encoded.validity().clone();
    let ptype = dtype.as_ptype();

    let decoded = match_each_alp_float_ptype!(ptype, |T| {
        let mut alp_buffer = encoded.into_buffer_mut();
        <T>::decode_slice_inplace(alp_buffer.as_mut_slice(), exponents);
        let decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };
        PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
    });

    if let Some(patches) = patches {
        decoded.patch(&patches)
    } else {
        decoded
    }
}
