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
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_vector::primitive::PVectorMut;

use crate::ALPArray;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// Decompresses an ALP-encoded array to a typed vector.
///
/// Uses chunked decompression (1024 elements) when patches have chunk offsets
/// for better L1 cache locality.
///
/// # Returns
///
/// A `PVectorMut<T>` with decompressed values and validity mask.
pub(crate) fn decompress_to_pvector<T: ALPFloat>(array: ALPArray) -> PVectorMut<T> {
    if array.is_empty() {
        return PVectorMut::with_capacity(0);
    }

    let (encoded, exponents, patches, dtype) = array.into_parts();
    let decompressed = if let Some(ref patches) = patches
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
    };

    let validity = decompressed.validity().clone();
    let buffer = decompressed.into_buffer_mut::<T>();
    let validity_mask = validity.to_mask(buffer.len()).into_mut();
    // SAFETY: buffer and validity_mask have same length.
    unsafe { PVectorMut::new_unchecked(buffer, validity_mask) }
}

/// Decompresses an ALP-encoded array.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn decompress(array: ALPArray) -> PrimitiveArray {
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
        PrimitiveArray::new::<T>(
            <T>::decode_buffer(encoded.into_buffer_mut(), exponents),
            validity,
        )
    });

    if let Some(patches) = patches {
        decoded.patch(&patches)
    } else {
        decoded
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_vector::VectorMutOps;

    use super::*;
    use crate::alp_encode;

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_vector_decompression_f32(#[case] size: usize) {
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f32));
        let encoded = alp_encode(&values, None).unwrap();
        let vector = decompress_to_pvector::<f32>(encoded);
        assert_eq!(vector.len(), size);
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_vector_decompression_f64(#[case] size: usize) {
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f64));
        let encoded = alp_encode(&values, None).unwrap();
        let vector = decompress_to_pvector::<f64>(encoded);
        assert_eq!(vector.len(), size);
    }

    #[rstest]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_vector_decompression_with_patches(#[case] size: usize) {
        use std::f64::consts::PI;

        let values: Vec<f64> = (0..size)
            .map(|i| match i % 4 {
                0..=2 => 1.0,
                _ => PI,
            })
            .collect();

        let array = PrimitiveArray::from_iter(values);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().unwrap().array_len() > 0);

        let vector = decompress_to_pvector::<f64>(encoded.clone());
        assert_eq!(vector.len(), size);

        let expected = decompress(encoded);
        assert_eq!(expected.as_slice::<f64>(), vector.as_ref());
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_vector_decompression_validity(#[case] size: usize) {
        let values: Vec<Option<f32>> = (0..size)
            .map(|i| if i % 2 == 1 { None } else { Some(1.0) })
            .collect();

        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(&array, None).unwrap();

        let vector = decompress_to_pvector::<f32>(encoded.clone());
        assert_eq!(vector.len(), size);

        let expected = decompress(encoded);

        assert_eq!(expected.as_slice::<f32>(), vector.as_ref());
        for i in 0..size {
            assert_eq!(expected.validity().is_valid(i), vector.validity().value(i));
        }
    }

    #[rstest]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_vector_decompression_with_patches_and_validity(#[case] size: usize) {
        use std::f64::consts::PI;

        let values: Vec<Option<f64>> = (0..size)
            .map(|i| match i % 3 {
                0 => Some(1.0),
                1 => None,
                _ => Some(PI),
            })
            .collect();

        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(&array, None).unwrap();
        assert!(encoded.patches().unwrap().array_len() > 0);

        let vector = decompress_to_pvector::<f64>(encoded.clone());
        assert_eq!(vector.len(), size);

        let expected = decompress(encoded);
        assert_eq!(expected.as_slice::<f64>(), vector.as_ref());

        for i in 0..size {
            assert_eq!(expected.validity().is_valid(i), vector.validity().value(i));
        }
    }
}
