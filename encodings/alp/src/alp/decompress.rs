// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::transmute;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patch_chunk;
use vortex_array::patches::Patches;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_vector::Vector;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PVectorMut;

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

/// Decompresses an ALP-encoded array.
///
/// # Returns
///
/// A `Vector` containing the decompressed floating-point values with all patches applied.
pub fn decompress_into_vector<T: ALPFloat>(
    encoded_vector: Vector,
    exponents: Exponents,
    patches_vectors: Option<(Vector, Vector, Option<Vector>)>,
    patches_offset: usize,
) -> VortexResult<Vector> {
    let encoded_primitive = encoded_vector.into_primitive().into_mut();
    let (mut alp_buffer, mask) = T::ALPInt::downcast(encoded_primitive).into_parts();
    <T>::decode_slice_inplace(alp_buffer.as_mut_slice(), exponents);

    // SAFETY: `Buffer<T::ALPInt> and `BufferMut<T>` have the same layout.
    let mut decoded_buffer: BufferMut<T> = unsafe { transmute(alp_buffer) };

    // Apply patches if they exist.
    if let Some((patches_indices, patches_values, _)) = patches_vectors {
        let patches_indices = patches_indices.into_primitive();
        let patches_values = patches_values.into_primitive();

        let values_buffer = T::downcast(patches_values.into_mut()).into_parts().0;
        let values_slice = values_buffer.as_slice();
        let decoded_slice = decoded_buffer.as_mut_slice();

        match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
            let indices_buffer = I::downcast(patches_indices.into_mut()).into_parts().0;
            let indices_slice = indices_buffer.as_slice();

            for (&idx, &value) in indices_slice.iter().zip(values_slice.iter()) {
                decoded_slice[AsPrimitive::<usize>::as_(idx) - patches_offset] = value;
            }
        });
    }

    Ok(PVectorMut::<T>::new(decoded_buffer, mask).freeze().into())
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
    let patches_offset = patches.offset();

    // We need to drop ALPArray here in case converting encoded buffer into
    // primitive didn't create a copy. In that case both alp_encoded and array
    // will hold a reference to the buffer we want to mutate.
    drop(array);

    match_each_alp_float_ptype!(ptype, |T| {
        let patches_values = patches_values.as_slice::<T>();
        let mut alp_buffer = encoded.into_buffer_mut();
        match_each_unsigned_integer_ptype!(patches_chunk_offsets.ptype(), |C| {
            let patches_chunk_offsets = patches_chunk_offsets.as_slice::<C>();
            // There always is at least one chunk offset.
            let base_offset = patches_chunk_offsets[0];
            let offset_within_chunk = patches.offset_within_chunk().unwrap_or(0);

            match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
                let patches_indices = patches_indices.as_slice::<I>();

                for (chunk_idx, chunk_start) in (0..array_len).step_by(1024).enumerate() {
                    let chunk_end = (chunk_start + 1024).min(array_len);
                    let chunk_slice = &mut alp_buffer.as_mut_slice()[chunk_start..chunk_end];

                    <T>::decode_slice_inplace(chunk_slice, exponents);

                    let decoded_chunk: &mut [T] = unsafe { transmute(chunk_slice) };
                    patch_chunk(
                        decoded_chunk,
                        patches_indices,
                        patches_values,
                        patches_offset,
                        patches_chunk_offsets,
                        chunk_idx,
                        base_offset.as_(),
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
