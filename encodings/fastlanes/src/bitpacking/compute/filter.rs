// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::FilterKernel;
use vortex_array::compute::FilterKernelAdapter;
use vortex_array::compute::filter;
use vortex_array::register_kernel;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::chunked_indices;
use crate::BitPackedArray;
use crate::BitPackedVTable;
use crate::bitpacking::compute::take::UNPACK_CHUNK_THRESHOLD;

impl FilterKernel for BitPackedVTable {
    fn filter(&self, array: &BitPackedArray, mask: &Mask) -> VortexResult<ArrayRef> {
        // Since the fastlanes crate only supports unsigned integers, and since we know that all
        // numbers are going to be non-negative, we can safely "cast" to unsigned.
        let ptype = array.ptype().to_unsigned();

        match_each_unsigned_integer_ptype!(ptype, |U| {
            // Note that the `filter_primitive` function will reinterpret cast the array back to the
            // correct `PType`, even if it was changed in `to_unsigned` above.
            Ok(filter_primitive::<U>(array, mask)?.into_array())
        })
    }
}

register_kernel!(FilterKernelAdapter(BitPackedVTable).lift());

/// Specialized filter kernel for primitive bit-packed arrays.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, the provided
/// `T` should be promoted to the unsigned variant for any target bit width.
/// For example, if the array is bit-packed `i16`, this function called be called with `T = u16`.
///
/// All bit-packing operations will use the unsigned kernels, but the logical type of `array`
/// dictates the final `PType` of the result.
///
/// This function fully decompresses the array for all but the most selective masks because the
/// FastLanes decompression is so fast and the bookkeepping necessary to decompress individual
/// elements is relatively slow. If you prefer to never fully decompress, use
/// [filter_primitive_no_decompression].
fn filter_primitive<T: NativePType + BitPacking>(
    array: &BitPackedArray,
    mask: &Mask,
) -> VortexResult<PrimitiveArray> {
    // Short-circuit if the selectivity is high enough.
    let full_decompression_threshold = match size_of::<T>() {
        1 => 0.03,
        2 => 0.03,
        4 => 0.075,
        _ => 0.09,
        // >8 bytes may have a higher threshold. These numbers are derived from a GCP c2-standard-4
        // with a "Cascade Lake" CPU.
    };
    if mask.density() >= full_decompression_threshold {
        let decompressed_array = array.to_primitive();
        Ok(filter(decompressed_array.as_ref(), mask)?.to_primitive())
    } else {
        filter_primitive_no_decompression::<T>(array, mask)
    }
}

/// Filter a bit-packed array, without using full decompression.
///
/// You should probably use [filter_primitive].
fn filter_primitive_no_decompression<T: NativePType + BitPacking>(
    array: &BitPackedArray,
    mask: &Mask,
) -> VortexResult<PrimitiveArray> {
    let validity = array.validity().filter(mask)?;

    let patches = array
        .patches()
        .map(|patches| patches.filter(mask))
        .transpose()?
        .flatten();

    let values: Buffer<T> = filter_indices(
        array,
        mask.true_count(),
        mask.values()
            .vortex_expect("AllTrue and AllFalse handled by filter fn")
            .indices()
            .iter()
            .copied(),
    );

    let mut values = PrimitiveArray::new(values, validity).reinterpret_cast(array.ptype());
    if let Some(patches) = patches {
        values = values.patch(&patches)?;
    }
    Ok(values)
}

fn filter_indices<T: NativePType + BitPacking>(
    array: &BitPackedArray,
    indices_len: usize,
    indices: impl Iterator<Item = usize>,
) -> Buffer<T> {
    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;
    let mut values = BufferMut::with_capacity(indices_len);

    // Some re-usable memory to store per-chunk indices.
    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; 1024];
    let packed_bytes = array.packed_slice::<T>();

    // Group the indices by the FastLanes chunk they belong to.
    let chunk_size = 128 * bit_width / size_of::<T>();

    chunked_indices(indices, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed_bytes[chunk_idx * chunk_size..][..chunk_size];

        if indices_within_chunk.len() == 1024 {
            // Unpack the entire chunk.
            unsafe {
                let values_len = values.len();
                values.set_len(values_len + 1024);
                BitPacking::unchecked_unpack(
                    bit_width,
                    packed,
                    &mut values.as_mut_slice()[values_len..],
                );
            }
        } else if indices_within_chunk.len() > UNPACK_CHUNK_THRESHOLD {
            // Unpack into a temporary chunk and then copy the values.
            unsafe {
                let dst: &mut [MaybeUninit<T>] = &mut unpacked;
                let dst: &mut [T] = mem::transmute(dst);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
            }
            values.extend_trusted(
                indices_within_chunk
                    .iter()
                    .map(|&idx| unsafe { unpacked.get_unchecked(idx).assume_init() }),
            );
        } else {
            // Otherwise, unpack each element individually.
            values.extend_trusted(indices_within_chunk.iter().map(|&idx| unsafe {
                BitPacking::unchecked_unpack_single(bit_width, packed, idx)
            }));
        }
    });

    values.freeze()
}

#[cfg(test)]
mod test {
    use vortex_array::Array;
    use vortex_array::IntoArray as _;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::filter;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = filter(bitpacked.as_ref(), &mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18])
        );
    }

    #[test]
    fn take_sliced_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = filter(&sliced, &mask).unwrap();
        assert_arrays_eq!(primitive_result, PrimitiveArray::from_iter([31u8, 33]));
    }

    #[test]
    fn filter_bitpacked() {
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(4096, (0..1024).collect()),
        )
        .unwrap();
        assert_arrays_eq!(
            filtered.to_primitive(),
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8))
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 9).unwrap();
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(values.len(), (0..250).collect()),
        )
        .unwrap()
        .to_primitive();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied())
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 3).unwrap();
        test_filter_conformance(bitpacked.as_ref());

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 9).unwrap();
        test_filter_conformance(bitpacked.as_ref());

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 3).unwrap();
        test_filter_conformance(bitpacked.as_ref());
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        // Create signed integer values where some exceed the bit width (causing patches).
        // Values 0-127 fit in 7 bits, but 1000 and 2000 do not.
        let values: Vec<i32> = vec![0, 10, 1000, 20, 30, 2000, 40, 50, 60, 70];
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 7).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Filter to include some patched and some non-patched values.
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(values.len(), vec![0, 2, 5, 9]),
        )
        .unwrap()
        .to_primitive();

        assert_arrays_eq!(filtered, PrimitiveArray::from_iter([0i32, 1000, 2000, 70]));
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        // Create a larger array with signed integers and some patches.
        let values: Vec<i32> = (0..1000)
            .map(|i| {
                if i % 100 == 0 {
                    10000 + i // These will be patches (exceed 7 bits)
                } else {
                    i % 128 // These fit in 7 bits
                }
            })
            .collect();
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 7).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Use low selectivity (only select 2% of values) to avoid full decompression.
        let indices: Vec<usize> = (0..20).collect();
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(values.len(), indices),
        )
        .unwrap()
        .to_primitive();

        let expected: Vec<i32> = values[0..20].to_vec();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected));
    }
}
