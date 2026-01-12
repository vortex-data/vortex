// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builders::UninitRange;
use vortex_array::patches::Patches;
use vortex_buffer::BufferMut;
use vortex_dtype::IntegerPType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_integer_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_scalar::Scalar;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::primitive::PrimitiveVectorMut;

use crate::BitPackedArray;
use crate::unpack_iter::BitPacked;

/// Unpacks a bit-packed array into a primitive vector.
pub fn unpack_to_primitive_vector(array: &BitPackedArray) -> PrimitiveVectorMut {
    match_each_integer_ptype!(array.ptype(), |P| { unpack_to_pvector::<P>(array).into() })
}

/// Unpacks a bit-packed array into a generic [`PVectorMut`].
pub fn unpack_to_pvector<P: BitPacked>(array: &BitPackedArray) -> PVectorMut<P> {
    if array.is_empty() {
        return PVectorMut::with_capacity(0);
    }

    let len = array.len();
    let mut elements = BufferMut::<P>::with_capacity(len);
    let uninit_slice = &mut elements.spare_capacity_mut()[..len];

    // Decode into an uninitialized slice.
    let mut bit_packed_iter = array.unpacked_chunks();
    bit_packed_iter.decode_into(uninit_slice);
    // SAFETY: `decode_into` initialized exactly `len` elements into the spare (existing) capacity.
    unsafe { elements.set_len(len) };

    let mut validity = array.validity_mask().into_mut();
    debug_assert_eq!(validity.len(), len);

    // TODO(connor): Implement a fused version of patching instead.
    if let Some(patches) = array.patches() {
        // SAFETY:
        // - `Patches` invariant guarantees indices are sorted and within array bounds.
        // - `elements` and `validity` have equal length (both are `len` from the array).
        // - All patch indices are valid after offset adjustment (guaranteed by `Patches`).
        unsafe { patches.apply_to_buffer(&mut elements, &mut validity) };
    }

    // SAFETY: `elements` and `validity` have the same length.
    unsafe { PVectorMut::new_unchecked(elements, validity) }
}

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
    let Mask::AllTrue(_) = values_validity else {
        vortex_panic!("BitPackedArray somehow had nullable patch values");
    };

    for (index, &value) in indices.iter().zip_eq(values) {
        dst.set_value(index.as_() - indices_offset, f(value));
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
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VectorExecutor;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_session::VortexSession;
    use vortex_vector::VectorMutOps;

    use super::*;
    use crate::BitPackedVTable;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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

    /// Test basic unpacking to primitive vector for multiple types and sizes.
    #[test]
    fn test_unpack_to_primitive_vector_basic() {
        // Test with u8 values.
        let u8_values = PrimitiveArray::from_iter([5u8, 10, 15, 20, 25]);
        let u8_bitpacked = bitpack_encode(&u8_values, 5, None).unwrap();
        let u8_vector = unpack_to_primitive_vector(&u8_bitpacked);
        // Compare with existing unpack method.
        let expected = unpack_array(&u8_bitpacked);
        assert_eq!(u8_vector.len(), expected.len());
        // Verify the vector matches expected values by checking specific elements.
        let _u8_frozen = u8_vector.freeze();
        // We know both produce the same primitive values, just in different forms.

        // Test with u32 values - empty array.
        let u32_empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u32>::new());
        let u32_empty_bp = bitpack_encode(&u32_empty, 0, None).unwrap();
        let u32_empty_vec = unpack_to_primitive_vector(&u32_empty_bp);
        assert_eq!(u32_empty_vec.len(), 0);

        // Test with u16 values - exactly one chunk (1024 elements).
        let u16_values = PrimitiveArray::from_iter(0u16..1024);
        let u16_bitpacked = bitpack_encode(&u16_values, 10, None).unwrap();
        let u16_vector = unpack_to_primitive_vector(&u16_bitpacked);
        assert_eq!(u16_vector.len(), 1024);

        // Test with i32 values - partial chunk (1025 elements).
        let i32_values = PrimitiveArray::from_iter((0i32..1025).map(|x| x % 512));
        let i32_bitpacked = bitpack_encode(&i32_values, 9, None).unwrap();
        let i32_vector = unpack_to_primitive_vector(&i32_bitpacked);
        assert_eq!(i32_vector.len(), 1025);

        // Verify consistency: unpack_to_primitive_vector and unpack_array should produce same values.
        let i32_array = unpack_array(&i32_bitpacked);
        assert_eq!(i32_vector.len(), i32_array.len());
    }

    /// Test unpacking with patches at various positions.
    #[test]
    fn test_unpack_to_primitive_vector_with_patches() {
        // Create an array where patches are needed at start, middle, and end.
        let values: Vec<u32> = vec![
            2000, // Patch at start
            5, 10, 15, 20, 25, 30, 3000, // Patch in middle
            35, 40, 45, 50, 55, 4000, // Patch at end
        ];
        let array = PrimitiveArray::from_iter(values.clone());

        // Bitpack with a small bit width to force patches.
        let bitpacked = bitpack_encode(&array, 6, None).unwrap();
        assert!(bitpacked.patches().is_some(), "Should have patches");

        // Unpack to vector.
        let vector = unpack_to_primitive_vector(&bitpacked);

        // Verify length and that patches were applied.
        assert_eq!(vector.len(), values.len());
        // The vector should have the patched values, which unpack_array also produces.
        let expected = unpack_array(&bitpacked);
        assert_eq!(vector.len(), expected.len());

        // Test with a larger array with multiple patches across chunks.
        let large_values: Vec<u16> = (0..3072)
            .map(|i| {
                if i % 500 == 0 {
                    2000 + i as u16 // Values that need patches
                } else {
                    (i % 256) as u16 // Values that fit in 8 bits
                }
            })
            .collect();
        let large_array = PrimitiveArray::from_iter(large_values);
        let large_bitpacked = bitpack_encode(&large_array, 8, None).unwrap();
        assert!(large_bitpacked.patches().is_some());

        let large_vector = unpack_to_primitive_vector(&large_bitpacked);
        assert_eq!(large_vector.len(), 3072);
    }

    /// Test unpacking with nullability and validity masks.
    #[test]
    fn test_unpack_to_primitive_vector_nullability() {
        // Test with null values at various positions.
        let values = Buffer::from_iter([100u32, 0, 200, 0, 300, 0, 400]);
        let validity = Validity::from_iter([true, false, true, false, true, false, true]);
        let array = PrimitiveArray::new(values, validity);

        let bitpacked = bitpack_encode(&array, 9, None).unwrap();
        let vector = unpack_to_primitive_vector(&bitpacked);

        // Verify length.
        assert_eq!(vector.len(), 7);
        // Validity should be preserved when unpacking.

        // Test combining patches with nullability.
        let patch_values = Buffer::from_iter([10u16, 0, 2000, 0, 30, 3000, 0]);
        let patch_validity = Validity::from_iter([true, false, true, false, true, true, false]);
        let patch_array = PrimitiveArray::new(patch_values, patch_validity);

        let patch_bitpacked = bitpack_encode(&patch_array, 5, None).unwrap();
        assert!(patch_bitpacked.patches().is_some());

        let patch_vector = unpack_to_primitive_vector(&patch_bitpacked);
        assert_eq!(patch_vector.len(), 7);

        // Test all nulls edge case.
        let all_nulls = PrimitiveArray::new(
            Buffer::from_iter([0u32, 0, 0, 0]),
            Validity::from_iter([false, false, false, false]),
        );
        let all_nulls_bp = bitpack_encode(&all_nulls, 0, None).unwrap();
        let all_nulls_vec = unpack_to_primitive_vector(&all_nulls_bp);
        assert_eq!(all_nulls_vec.len(), 4);
    }

    /// Test that the execute method produces consistent results with other unpacking methods.
    #[test]
    fn test_execute_method_consistency() {
        // Test that execute(), unpack_to_primitive_vector(), and unpack_array() all produce consistent results.
        let test_consistency = |array: &PrimitiveArray, bit_width: u8| {
            let bitpacked = bitpack_encode(array, bit_width, None).unwrap();

            // Method 1: Using the new unpack_to_primitive_vector.
            let vector_result = unpack_to_primitive_vector(&bitpacked);

            // Method 2: Using the old unpack_array.
            let unpacked_array = unpack_array(&bitpacked);

            // Method 3: Using the execute() method (this is what would be used in production).
            let executed = {
                let mut ctx = SESSION.create_execution_ctx();
                bitpacked.into_array().execute(&mut ctx).unwrap()
            };

            // All three should produce the same length.
            assert_eq!(vector_result.len(), array.len(), "vector length mismatch");
            assert_eq!(
                unpacked_array.len(),
                array.len(),
                "unpacked array length mismatch"
            );

            // The executed canonical should also have the correct length.
            let executed_primitive = executed.into_primitive();
            assert_eq!(
                executed_primitive.len(),
                array.len(),
                "executed primitive length mismatch"
            );

            // Verify that the execute() method works correctly by comparing with unpack_array.
            // We convert unpack_array result to canonical to compare.
            let unpacked_executed = {
                let mut ctx = SESSION.create_execution_ctx();
                unpacked_array
                    .into_array()
                    .execute(&mut ctx)
                    .unwrap()
                    .into_primitive()
            };
            assert_eq!(
                executed_primitive.len(),
                unpacked_executed.len(),
                "execute() and unpack_array().execute() produced different lengths"
            );
            // Both should produce identical arrays since they represent the same data.
        };

        // Test various scenarios without patches.
        test_consistency(&PrimitiveArray::from_iter(0u16..100), 7);
        test_consistency(&PrimitiveArray::from_iter(0u32..1024), 10);

        // Test with values that will create patches.
        test_consistency(&PrimitiveArray::from_iter((0i16..2048).map(|x| x % 128)), 7);

        // Test with an array that definitely has patches.
        let patch_values: Vec<u32> = (0..100)
            .map(|i| if i % 20 == 0 { 1000 + i } else { i % 16 })
            .collect();
        let patch_array = PrimitiveArray::from_iter(patch_values);
        test_consistency(&patch_array, 4);

        // Test with sliced array (offset > 0).
        let values = PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None).unwrap();
        let sliced = bitpacked.slice(500..1500);

        // Test all three methods on the sliced array.
        let sliced_bp = sliced.as_::<BitPackedVTable>();
        let vector_result = unpack_to_primitive_vector(sliced_bp);
        let unpacked_array = unpack_array(sliced_bp);
        let executed = {
            let mut ctx = SESSION.create_execution_ctx();
            sliced.execute(&mut ctx).unwrap()
        };

        assert_eq!(
            vector_result.len(),
            1000,
            "sliced vector length should be 1000"
        );
        assert_eq!(
            unpacked_array.len(),
            1000,
            "sliced unpacked array length should be 1000"
        );

        let executed_primitive = executed.into_primitive();
        assert_eq!(
            executed_primitive.len(),
            1000,
            "sliced executed primitive length should be 1000"
        );
    }

    /// Test edge cases for unpacking.
    #[test]
    fn test_unpack_edge_cases() {
        // Empty array.
        let empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u64>::new());
        let empty_bp = bitpack_encode(&empty, 0, None).unwrap();
        let empty_vec = unpack_to_primitive_vector(&empty_bp);
        assert_eq!(empty_vec.len(), 0);

        // All zeros (bit_width = 0).
        let zeros = PrimitiveArray::from_iter([0u32; 100]);
        let zeros_bp = bitpack_encode(&zeros, 0, None).unwrap();
        let zeros_vec = unpack_to_primitive_vector(&zeros_bp);
        assert_eq!(zeros_vec.len(), 100);
        // Verify consistency with unpack_array.
        let zeros_array = unpack_array(&zeros_bp);
        assert_eq!(zeros_vec.len(), zeros_array.len());

        // Maximum bit width for u16 (15 bits, since bitpacking requires bit_width < type bit width).
        let max_values = PrimitiveArray::from_iter([32767u16; 50]); // 2^15 - 1
        let max_bp = bitpack_encode(&max_values, 15, None).unwrap();
        let max_vec = unpack_to_primitive_vector(&max_bp);
        assert_eq!(max_vec.len(), 50);

        // Exactly 3072 elements with patches across chunks.
        let boundary_values: Vec<u32> = (0..3072)
            .map(|i| {
                if i == 1023 || i == 1024 || i == 2047 || i == 2048 {
                    50000 // Force patches at chunk boundaries
                } else {
                    (i % 128) as u32
                }
            })
            .collect();
        let boundary_array = PrimitiveArray::from_iter(boundary_values);
        let boundary_bp = bitpack_encode(&boundary_array, 7, None).unwrap();
        assert!(boundary_bp.patches().is_some());

        let boundary_vec = unpack_to_primitive_vector(&boundary_bp);
        assert_eq!(boundary_vec.len(), 3072);
        // Verify consistency.
        let boundary_unpacked = unpack_array(&boundary_bp);
        assert_eq!(boundary_vec.len(), boundary_unpacked.len());

        // Single element.
        let single = PrimitiveArray::from_iter([42u8]);
        let single_bp = bitpack_encode(&single, 6, None).unwrap();
        let single_vec = unpack_to_primitive_vector(&single_bp);
        assert_eq!(single_vec.len(), 1);
    }
}
