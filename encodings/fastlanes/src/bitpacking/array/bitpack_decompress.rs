// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder, UninitRange};
use vortex_array::patches::Patches;
use vortex_dtype::{
    IntegerPType, NativePType, match_each_integer_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::BitPackedArray;
use crate::unpack_iter::BitPacked;

pub fn unpack_array(array: &BitPackedArray) -> PrimitiveArray {
    match_each_integer_ptype!(array.ptype(), |P| { unpack_primitive_array::<P>(array) })
}

pub fn unpack_primitive_array<T: BitPacked>(array: &BitPackedArray) -> PrimitiveArray {
    let mut builder = PrimitiveBuilder::with_capacity(array.dtype().nullability(), array.len());
    unpack_into_primitive_builder::<T>(array, &mut builder);
    assert_eq!(builder.len(), array.len());
    builder.finish_into_primitive()
}

pub(crate) fn unpack_into_primitive_builder<T: BitPacked>(
    array: &BitPackedArray,
    // TODO(ngates): do we want to use fastlanes alignment for this buffer?
    builder: &mut PrimitiveBuilder<T>,
) {
    // If the array is empty, then we don't need to add anything to the builder.
    if array.is_empty() {
        return;
    }

    let mut uninit_range = builder.uninit_range(array.len());

    // SAFETY: We later initialize the the uninitialized range of values with `copy_from_slice`.
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(array.validity_mask());
    }

    // SAFETY: `decode_into` will initialize all values in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, array.len()) };

    let mut bit_packed_iter = array.unpacked_chunks();
    bit_packed_iter.decode_into(uninit_slice);

    if let Some(patches) = array.patches() {
        apply_patches_to_uninit_range(&mut uninit_range, patches);
    };

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via `decode_into`.
    unsafe {
        uninit_range.finish();
    }
}

pub fn apply_patches_to_uninit_range<T: NativePType>(dst: &mut UninitRange<T>, patches: &Patches) {
    apply_patches_to_uninit_range_fn(dst, patches, |x| x)
}

pub fn apply_patches_to_uninit_range_fn<T: NativePType, F: Fn(T) -> T>(
    dst: &mut UninitRange<T>,
    patches: &Patches,
    f: F,
) {
    assert_eq!(patches.array_len(), dst.len());

    let indices = patches.indices().to_primitive();
    let values = patches.values().to_primitive();
    let validity = values.validity_mask();
    let values = values.as_slice::<T>();

    match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        insert_values_and_validity_at_indices_to_uninit_range(
            dst,
            indices.as_slice::<P>(),
            values,
            validity,
            patches.offset(),
            f,
        )
    });
}

fn insert_values_and_validity_at_indices_to_uninit_range<
    T: NativePType,
    IndexT: IntegerPType,
    F: Fn(T) -> T,
>(
    dst: &mut UninitRange<T>,
    indices: &[IndexT],
    values: &[T],
    values_validity: Mask,
    indices_offset: usize,
    f: F,
) {
    match values_validity {
        Mask::AllTrue(_) => {
            for (index, &value) in indices.iter().zip_eq(values) {
                dst.set_value(index.as_() - indices_offset, f(value));
            }
        }
        Mask::AllFalse(_) => {
            for decompressed_index in indices {
                dst.set_validity_bit(decompressed_index.as_() - indices_offset, false);
            }
        }
        Mask::Values(vb) => {
            for (index, &value) in indices.iter().zip_eq(values) {
                let out_index = index.as_() - indices_offset;
                dst.set_value(out_index, f(value));
                dst.set_validity_bit(out_index, vb.value(out_index));
            }
        }
    }
}

pub fn unpack_single(array: &BitPackedArray, index: usize) -> Scalar {
    let bit_width = array.bit_width() as usize;
    let ptype = array.ptype();
    // let packed = array.packed().into_primitive()?;
    let index_in_encoded = index + array.offset() as usize;
    let scalar: Scalar = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |P| {
        unsafe {
            unpack_single_primitive::<P>(array.packed_slice::<P>(), bit_width, index_in_encoded)
                .into()
        }
    });
    // Cast to fix signedness and nullability
    scalar.cast(array.dtype()).vortex_expect("cast failure")
}

/// # Safety
///
/// The caller must ensure the following invariants hold:
/// * `packed.len() == (length + 1023) / 1024 * 128 * bit_width`
/// * `index_to_decode < length`
///
/// Where `length` is the length of the array/slice backed by `packed`
/// (but is not provided to this function).
pub unsafe fn unpack_single_primitive<T: NativePType + BitPacking>(
    packed: &[T],
    bit_width: usize,
    index_to_decode: usize,
) -> T {
    let chunk_index = index_to_decode / 1024;
    let index_in_chunk = index_to_decode % 1024;
    let elems_per_chunk: usize = 128 * bit_width / size_of::<T>();

    let packed_chunk = &packed[chunk_index * elems_per_chunk..][0..elems_per_chunk];
    unsafe { BitPacking::unchecked_unpack_single(bit_width, packed_chunk, index_in_chunk) }
}

pub fn count_exceptions(bit_width: u8, bit_width_freq: &[usize]) -> usize {
    if bit_width_freq.len() <= bit_width as usize {
        return 0;
    }
    bit_width_freq[bit_width as usize + 1..].iter().sum()
}

#[cfg(test)]
mod tests {
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, assert_arrays_eq};
    use vortex_buffer::{Buffer, BufferMut, buffer};
    use vortex_dtype::Nullability;

    use super::*;
    use crate::BitPackedVTable;
    use crate::bitpack_compress::bitpack_encode;

    fn compression_roundtrip(n: usize) {
        let values = PrimitiveArray::from_iter((0..n).map(|i| (i % 2047) as u16));
        let compressed = BitPackedArray::encode(values.as_ref(), 11).unwrap();
        let decompressed = compressed.to_primitive();
        assert_arrays_eq!(decompressed, values);

        values
            .as_slice::<u16>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                let scalar: u16 = unpack_single(&compressed, i).try_into().unwrap();
                assert_eq!(scalar, *v);
            });
    }

    #[test]
    fn test_compression_roundtrip_fast() {
        compression_roundtrip(125);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn test_compression_roundtrip() {
        compression_roundtrip(1024);
        compression_roundtrip(10_000);
        compression_roundtrip(10_240);
    }

    #[test]
    fn test_all_zeros() {
        let zeros = buffer![0u16, 0, 0, 0].into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 0, None).unwrap();
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([0u16, 0, 0, 0]));
    }

    #[test]
    fn test_simple_patches() {
        let zeros = buffer![0u16, 1, 0, 1].into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 0, None).unwrap();
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([0u16, 1, 0, 1]));
    }

    #[test]
    fn test_one_full_chunk() {
        let zeros = BufferMut::from_iter(0u16..1024).into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(0u16..1024));
    }

    #[test]
    fn test_three_full_chunks_with_patches() {
        let zeros = BufferMut::from_iter((5u16..1029).chain(5u16..1029).chain(5u16..1029))
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert!(bitpacked.patches().is_some());
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter((5u16..1029).chain(5u16..1029).chain(5u16..1029))
        );
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_no_patch() {
        let zeros = BufferMut::from_iter(0u16..1025).into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 11, None).unwrap();
        assert!(bitpacked.patches().is_none());
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(0u16..1025));
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_with_patches() {
        let zeros = BufferMut::from_iter(512u16..1537)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 1025);
        assert!(bitpacked.patches().is_some());
        let actual = unpack_array(&bitpacked);
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(512u16..1537));
    }

    #[test]
    fn test_offset_and_short_chunk_and_patches() {
        let zeros = BufferMut::from_iter(512u16..1537)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 1025);
        assert!(bitpacked.patches().is_some());
        let bitpacked = bitpacked.slice(1023..1025);
        let actual = unpack_array(bitpacked.as_::<BitPackedVTable>());
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([1535u16, 1536]));
    }

    #[test]
    fn test_offset_and_short_chunk_with_chunks_between_and_patches() {
        let zeros = BufferMut::from_iter(512u16..2741)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 2229);
        assert!(bitpacked.patches().is_some());
        let bitpacked = bitpacked.slice(1023..2049);
        let actual = unpack_array(bitpacked.as_::<BitPackedVTable>());
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter((1023u16..2049).map(|x| x + 512))
        );
    }

    #[test]
    fn test_unpack_into_empty_array() {
        let empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u32>::new());
        let bitpacked = bitpack_encode(&empty, 0, None).unwrap();

        let mut builder = PrimitiveBuilder::<u32>::new(Nullability::NonNullable);
        unpack_into_primitive_builder(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();
        assert_eq!(
            result.len(),
            0,
            "Empty array should result in empty builder"
        );
    }

    /// This test ensures that the mask is properly appended to the range, not the builder.
    #[test]
    fn test_unpack_into_with_validity_mask() {
        // Create an array with some null values.
        let values = Buffer::from_iter([1u32, 0, 3, 4, 0]);
        let validity = Validity::from_iter([true, false, true, true, false]);
        let array = PrimitiveArray::new(values, validity);

        // Bitpack the array.
        let bitpacked = bitpack_encode(&array, 3, None).unwrap();

        // Unpack into a new builder.
        let mut builder = PrimitiveBuilder::<u32>::with_capacity(Nullability::Nullable, 5);
        unpack_into_primitive_builder(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();

        // Verify the validity mask was correctly applied.
        assert_eq!(result.len(), 5);
        assert!(!result.scalar_at(0).is_null());
        assert!(result.scalar_at(1).is_null());
        assert!(!result.scalar_at(2).is_null());
        assert!(!result.scalar_at(3).is_null());
        assert!(result.scalar_at(4).is_null());
    }

    /// Test that `unpack_into` correctly handles arrays with patches.
    #[test]
    fn test_unpack_into_with_patches() {
        // Create an array where most values fit in 4 bits but some need patches.
        let values: Vec<u32> = (0..100)
            .map(|i| if i % 20 == 0 { 1000 + i } else { i % 16 })
            .collect();
        let array = PrimitiveArray::from_iter(values.clone());

        // Bitpack with a bit width that will require patches.
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Should have patches for values > 15"
        );

        // Unpack into a new builder.
        let mut builder = PrimitiveBuilder::<u32>::with_capacity(Nullability::NonNullable, 100);
        unpack_into_primitive_builder(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();

        // Verify all values were correctly unpacked including patches.
        assert_arrays_eq!(result, PrimitiveArray::from_iter(values));
    }
}
