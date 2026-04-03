// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builders::UninitRange;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::BitPackedData;
use crate::unpack_iter::BitPacked;

/// Unpacks a bit-packed array into a primitive array.
pub fn unpack_array(array: &BitPackedData, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.ptype(), |P| {
        unpack_primitive_array::<P>(array, ctx)
    })
}

pub fn unpack_primitive_array<T: BitPacked>(
    array: &BitPackedData,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mut builder = PrimitiveBuilder::with_capacity(array.dtype.nullability(), array.len);
    unpack_into_primitive_builder::<T>(array, &mut builder)?;
    assert_eq!(builder.len(), array.len);
    Ok(builder.finish_into_primitive())
}

pub(crate) fn unpack_into_primitive_builder<T: BitPacked>(
    array: &BitPackedData,
    // TODO(ngates): do we want to use fastlanes alignment for this buffer?
    builder: &mut PrimitiveBuilder<T>,
) -> VortexResult<()> {
    // If the array is empty, then we don't need to add anything to the builder.
    if array.len == 0 {
        return Ok(());
    }

    let mut uninit_range = builder.uninit_range(array.len);

    // SAFETY: We later initialize the the uninitialized range of values with `copy_from_slice`.
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(array.validity().to_mask(array.len));
    }

    // SAFETY: `decode_into` will initialize all values in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, array.len) };

    let mut bit_packed_iter = array.unpacked_chunks();
    bit_packed_iter.decode_into(uninit_slice);

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via `decode_into`.
    unsafe {
        uninit_range.finish();
    }
    Ok(())
}

pub fn apply_patches_to_uninit_range<T: NativePType>(
    dst: &mut UninitRange<T>,
    patches: &Patches,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    apply_patches_to_uninit_range_fn(dst, patches, ctx, |x| x)
}

pub fn apply_patches_to_uninit_range_fn<T: NativePType, F: Fn(T) -> T>(
    dst: &mut UninitRange<T>,
    patches: &Patches,
    ctx: &mut ExecutionCtx,
    f: F,
) -> VortexResult<()> {
    assert_eq!(patches.array_len(), dst.len());

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = values.validity_mask()?;
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
    Ok(())
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

pub fn unpack_single(array: &BitPackedData, index: usize) -> Scalar {
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
    scalar.cast(&array.dtype).vortex_expect("cast failure")
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

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedArray;
    use crate::bitpack_compress::BitPackedEncoder;

    fn encode(array: &PrimitiveArray, bit_width: u8) -> BitPackedArray {
        BitPackedEncoder::new(array)
            .with_bit_width(bit_width)
            .pack()
            .unwrap()
            .into_packed()
    }

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn compression_roundtrip(n: usize) {
        let values = PrimitiveArray::from_iter((0..n).map(|i| (i % 2047) as u16));
        let compressed = BitPackedEncoder::new(&values)
            .with_bit_width(11)
            .pack()
            .unwrap()
            .unwrap_unpatched();
        assert_arrays_eq!(compressed, values);

        values
            .as_slice::<u16>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                let scalar: u16 = (&unpack_single(&compressed, i)).try_into().unwrap();
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
    fn test_all_zeros() -> VortexResult<()> {
        let zeros = buffer![0u16, 0, 0, 0].into_array().to_primitive();
        let bitpacked = encode(&zeros, 0);
        let actual = unpack_array(&bitpacked, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([0u16, 0, 0, 0]));
        Ok(())
    }

    #[test]
    fn test_simple_patches() -> VortexResult<()> {
        let zeros = buffer![0u16, 1, 0, 1].into_array().to_primitive();
        let bitpacked = encode(&zeros, 0);
        let actual = unpack_array(&bitpacked, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([0u16, 1, 0, 1]));
        Ok(())
    }

    #[test]
    fn test_one_full_chunk() -> VortexResult<()> {
        let values = BufferMut::from_iter(0u16..1024).into_array().to_primitive();
        let bitpacked = encode(&values, 10);
        let actual = unpack_array(&bitpacked, &mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(0u16..1024));
        Ok(())
    }

    #[test]
    fn test_three_full_chunks_with_patches() -> VortexResult<()> {
        let values = BufferMut::from_iter((5u16..1029).chain(5u16..1029).chain(5u16..1029))
            .into_array()
            .to_primitive();
        let packed = BitPackedEncoder::new(&values).with_bit_width(10).pack()?;
        assert!(packed.has_patches());
        let actual = packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter((5u16..1029).chain(5u16..1029).chain(5u16..1029))
        );
        Ok(())
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_no_patch() -> VortexResult<()> {
        let values = BufferMut::from_iter(0u16..1025).into_array().to_primitive();
        let packed = BitPackedEncoder::new(&values).with_bit_width(11).pack()?;
        assert!(!packed.has_patches());
        let actual = packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(0u16..1025));
        Ok(())
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_with_patches() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(512u16..1537);
        let packed = BitPackedEncoder::new(&values).with_bit_width(10).pack()?;
        let bitpacked = packed.into_array()?;
        assert_eq!(bitpacked.len(), 1025);
        let actual = bitpacked.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(actual, PrimitiveArray::from_iter(512u16..1537));
        Ok(())
    }

    #[test]
    fn test_offset_and_short_chunk_and_patches() -> VortexResult<()> {
        let values = BufferMut::from_iter(512u16..1537)
            .into_array()
            .to_primitive();
        let packed = BitPackedEncoder::new(&values).with_bit_width(10).pack()?;
        assert!(packed.has_patches());
        let bitpacked = packed.into_array()?;
        assert_eq!(bitpacked.len(), 1025);
        let slice_ref = bitpacked.slice(1023..1025)?;
        let actual = {
            let mut ctx = SESSION.create_execution_ctx();
            slice_ref.execute::<Canonical>(&mut ctx)?.into_primitive()
        };
        assert_arrays_eq!(actual, PrimitiveArray::from_iter([1535u16, 1536]));
        Ok(())
    }

    #[test]
    fn test_offset_and_short_chunk_with_chunks_between_and_patches() -> VortexResult<()> {
        let values = BufferMut::from_iter(512u16..2741)
            .into_array()
            .to_primitive();
        let packed = BitPackedEncoder::new(&values).with_bit_width(10).pack()?;
        assert!(packed.has_patches());
        let bitpacked = packed.into_array()?;
        assert_eq!(bitpacked.len(), 2229);
        let slice_ref = bitpacked.into_array().slice(1023..2049)?;
        let actual = {
            let mut ctx = SESSION.create_execution_ctx();
            slice_ref.execute::<Canonical>(&mut ctx)?.into_primitive()
        };
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter((1023u16..2049).map(|x| x + 512))
        );
        Ok(())
    }

    #[test]
    fn test_unpack_into_empty_array() -> VortexResult<()> {
        let empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u32>::new());
        let bitpacked = encode(&empty, 0);

        let mut builder = PrimitiveBuilder::<u32>::new(Nullability::NonNullable);
        unpack_into_primitive_builder(&bitpacked, &mut builder)?;

        let result = builder.finish_into_primitive();
        assert_eq!(
            result.len(),
            0,
            "Empty array should result in empty builder"
        );
        Ok(())
    }

    /// This test ensures that the mask is properly appended to the range, not the builder.
    #[test]
    fn test_unpack_into_with_validity_mask() -> VortexResult<()> {
        // Create an array with some null values.
        let values = Buffer::from_iter([1u32, 0, 3, 4, 0]);
        let validity = Validity::from_iter([true, false, true, true, false]);
        let array = PrimitiveArray::new(values, validity);

        // Bitpack the array.
        let bitpacked = encode(&array, 3);

        // Unpack into a new builder.
        let mut builder = PrimitiveBuilder::<u32>::with_capacity(Nullability::Nullable, 5);
        unpack_into_primitive_builder(&bitpacked, &mut builder)?;

        let result = builder.finish_into_primitive();

        // Verify the validity mask was correctly applied.
        assert_eq!(result.len(), 5);
        assert!(!result.scalar_at(0)?.is_null());
        assert!(result.scalar_at(1)?.is_null());
        assert!(!result.scalar_at(2)?.is_null());
        assert!(!result.scalar_at(3)?.is_null());
        assert!(result.scalar_at(4)?.is_null());
        Ok(())
    }

    /// Test basic unpacking to primitive array for multiple types and sizes.
    #[test]
    fn test_execute_basic() -> VortexResult<()> {
        // Test with u8 values.
        let u8_values = PrimitiveArray::from_iter([5u8, 10, 15, 20, 25]);
        let u8_bitpacked = BitPackedEncoder::new(&u8_values)
            .with_bit_width(5)
            .pack()?
            .into_array()?;
        let u8_result =
            u8_bitpacked.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(u8_result.len(), 5);
        assert_arrays_eq!(u8_result, u8_values);

        // Test with u32 values - empty array.
        let u32_empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u32>::new());
        let u32_empty_bp = BitPackedEncoder::new(&u32_empty)
            .with_bit_width(0)
            .pack()?
            .into_array()?;
        let u32_empty_result =
            u32_empty_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(u32_empty_result.len(), 0);

        // Test with u16 values - exactly one chunk (1024 elements).
        let u16_values = PrimitiveArray::from_iter(0u16..1024);
        let u16_bitpacked = BitPackedEncoder::new(&u16_values)
            .with_bit_width(10)
            .pack()?
            .into_array()?;
        let u16_result =
            u16_bitpacked.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(u16_result.len(), 1024);

        // Test with i32 values - partial chunk (1025 elements).
        let i32_values = PrimitiveArray::from_iter((0i32..1025).map(|x| x % 512));
        let i32_bitpacked = BitPackedEncoder::new(&i32_values)
            .with_bit_width(9)
            .pack()?
            .into_array()?;
        let i32_result =
            i32_bitpacked.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(i32_result.len(), 1025);
        assert_arrays_eq!(i32_result, i32_values);
        Ok(())
    }

    /// Test unpacking with patches at various positions.
    #[test]
    fn test_execute_with_patches() -> VortexResult<()> {
        // Create an array where patches are needed at start, middle, and end.
        let values: Vec<u32> = vec![
            2000, // Patch at start
            5, 10, 15, 20, 25, 30, 3000, // Patch in middle
            35, 40, 45, 50, 55, 4000, // Patch at end
        ];
        let array = PrimitiveArray::from_iter(values.clone());

        // Bitpack with a small bit width to force patches.
        let packed = BitPackedEncoder::new(&array).with_bit_width(6).pack()?;
        assert!(packed.has_patches(), "Should have patches");

        // Execute to primitive array.
        let result = packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;

        // Verify length and values.
        assert_eq!(result.len(), values.len());
        assert_arrays_eq!(result, PrimitiveArray::from_iter(values));

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
        let large_array = PrimitiveArray::from_iter(large_values.clone());
        let large_packed = BitPackedEncoder::new(&large_array)
            .with_bit_width(8)
            .pack()?;
        assert!(large_packed.has_patches());

        let large_result = large_packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(large_result.len(), 3072);
        assert_arrays_eq!(large_result, PrimitiveArray::from_iter(large_values));
        Ok(())
    }

    /// Test unpacking with nullability and validity masks.
    #[test]
    fn test_execute_nullability() -> VortexResult<()> {
        // Test with null values at various positions.
        let values = Buffer::from_iter([100u32, 0, 200, 0, 300, 0, 400]);
        let validity = Validity::from_iter([true, false, true, false, true, false, true]);
        let array = PrimitiveArray::new(values, validity);

        let bitpacked = BitPackedEncoder::new(&array)
            .with_bit_width(9)
            .pack()?
            .into_array()?;
        let result = bitpacked.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;

        // Verify length.
        assert_eq!(result.len(), 7);
        // Validity should be preserved when unpacking.
        assert!(!result.scalar_at(0)?.is_null());
        assert!(result.scalar_at(1)?.is_null());
        assert!(!result.scalar_at(2)?.is_null());

        // Test combining patches with nullability.
        let patch_values = Buffer::from_iter([10u16, 0, 2000, 0, 30, 3000, 0]);
        let patch_validity = Validity::from_iter([true, false, true, false, true, true, false]);
        let patch_array = PrimitiveArray::new(patch_values, patch_validity);

        let patch_packed = BitPackedEncoder::new(&patch_array)
            .with_bit_width(5)
            .pack()?;
        assert!(patch_packed.has_patches());

        let patch_result = patch_packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(patch_result.len(), 7);

        // Test all nulls edge case.
        let all_nulls = PrimitiveArray::new(
            Buffer::from_iter([0u32, 0, 0, 0]),
            Validity::from_iter([false, false, false, false]),
        );
        let all_nulls_bp = BitPackedEncoder::new(&all_nulls)
            .with_bit_width(0)
            .pack()?
            .into_array()?;
        let all_nulls_result =
            all_nulls_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(all_nulls_result.len(), 4);
        Ok(())
    }

    /// Test that the execute method produces consistent results.
    #[test]
    fn test_execute_method_consistency() -> VortexResult<()> {
        let test_consistency = |array: &PrimitiveArray, bit_width: u8| -> VortexResult<()> {
            let packed = BitPackedEncoder::new(array)
                .with_bit_width(bit_width)
                .pack()?;

            // Using the execute() method.
            let executed = {
                let mut ctx = SESSION.create_execution_ctx();
                packed.into_array()?.execute::<Canonical>(&mut ctx).unwrap()
            };

            // The executed canonical should have the correct length.
            let executed_primitive = executed.into_primitive();
            assert_eq!(
                executed_primitive.len(),
                array.len(),
                "executed primitive length mismatch"
            );
            Ok(())
        };

        // Test various scenarios without patches.
        test_consistency(&PrimitiveArray::from_iter(0u16..100), 7)?;
        test_consistency(&PrimitiveArray::from_iter(0u32..1024), 10)?;

        // Test with values that will create patches.
        test_consistency(&PrimitiveArray::from_iter((0i16..2048).map(|x| x % 128)), 7)?;

        // Test with an array that definitely has patches.
        let patch_values: Vec<u32> = (0..100)
            .map(|i| if i % 20 == 0 { 1000 + i } else { i % 16 })
            .collect();
        let patch_array = PrimitiveArray::from_iter(patch_values);
        test_consistency(&patch_array, 4)?;

        // Test with sliced array (offset > 0).
        let values = PrimitiveArray::from_iter(0u32..2048);
        let packed = BitPackedEncoder::new(&values).with_bit_width(11).pack()?;
        let slice_ref = packed.into_array()?.slice(500..1500)?;
        let sliced = {
            let mut ctx = SESSION.create_execution_ctx();
            slice_ref.execute::<Canonical>(&mut ctx)?.into_primitive()
        };

        assert_eq!(sliced.len(), 1000, "sliced primitive length should be 1000");
        Ok(())
    }

    /// Test edge cases for unpacking.
    #[test]
    fn test_execute_edge_cases() -> VortexResult<()> {
        // Empty array.
        let empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u64>::new());
        let empty_bp = BitPackedEncoder::new(&empty)
            .with_bit_width(0)
            .pack()?
            .into_array()?;
        let empty_result =
            empty_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(empty_result.len(), 0);

        // All zeros (bit_width = 0).
        let zeros = PrimitiveArray::from_iter([0u32; 100]);
        let zeros_bp = BitPackedEncoder::new(&zeros)
            .with_bit_width(0)
            .pack()?
            .into_array()?;
        let zeros_result =
            zeros_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(zeros_result.len(), 100);
        assert_arrays_eq!(zeros_result, zeros);

        // Maximum bit width for u16 (15 bits, since bitpacking requires bit_width < type bit width).
        let max_values = PrimitiveArray::from_iter([32767u16; 50]); // 2^15 - 1
        let max_bp = BitPackedEncoder::new(&max_values)
            .with_bit_width(15)
            .pack()?
            .into_array()?;
        let max_result = max_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(max_result.len(), 50);

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
        let boundary_array = PrimitiveArray::from_iter(boundary_values.clone());
        let boundary_packed = BitPackedEncoder::new(&boundary_array)
            .with_bit_width(7)
            .pack()?;
        assert!(boundary_packed.has_patches());

        let boundary_result = boundary_packed
            .into_array()?
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(boundary_result.len(), 3072);
        assert_arrays_eq!(boundary_result, PrimitiveArray::from_iter(boundary_values));

        // Single element.
        let single = PrimitiveArray::from_iter([42u8]);
        let single_bp = BitPackedEncoder::new(&single)
            .with_bit_width(6)
            .pack()?
            .into_array()?;
        let single_result =
            single_bp.execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(single_result.len(), 1);
        Ok(())
    }
}
