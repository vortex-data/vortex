// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::sync::Arc;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use super::chunked_indices;
use super::take::UNPACK_CHUNK_THRESHOLD;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

/// The threshold over which it is faster to fully unpack the entire [`BitPackedArray`] and then
/// filter the result than to unpack only specific bitpacked values into the output buffer.
pub const fn unpack_then_filter_threshold(ptype: PType) -> f64 {
    // TODO(connor): Where did these numbers come from? Add a public link after validating them.
    // These numbers probably don't work for in-place filtering either.
    match ptype.byte_width() {
        1 => 0.03,
        2 => 0.03,
        4 => 0.075,
        _ => 0.09,
        // >8 bytes may have a higher threshold. These numbers are derived from a GCP c2-standard-4
        // with a "Cascade Lake" CPU.
    }
}

/// Kernel to execute filtering directly on a bit-packed array.
impl FilterKernel for BitPacked {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => {
                return Ok(None);
            }
            Mask::Values(values) => values,
        };

        // If the density is high enough, then we would rather decompress the whole array and then apply
        // a filter over decompressing values one by one.
        if values.density() > unpack_then_filter_threshold(array.dtype().as_ptype()) {
            return Ok(None);
        }

        // Filter and patch using the correct unsigned type for FastLanes, then cast to signed if needed.
        let primitive =
            match_each_unsigned_integer_ptype!(array.dtype().as_ptype().to_unsigned(), |U| {
                let (buffer, validity) = filter_primitive_without_patches::<U>(array, values)?;
                // reinterpret_cast for signed types.
                let primitive = PrimitiveArray::new(buffer, validity);
                if array.dtype().as_ptype().is_signed_int() {
                    PrimitiveArray::from_buffer_handle(
                        primitive.buffer_handle().clone(),
                        array.dtype().as_ptype(),
                        primitive.validity()?,
                    )
                } else {
                    primitive
                }
            });

        let patches = array
            .patches()
            .map(|patches| patches.filter(&Mask::Values(Arc::clone(values)), ctx))
            .transpose()?
            .flatten();

        if let Some(patches) = patches {
            let mut prim_array = primitive;
            prim_array = prim_array.patch(&patches, ctx)?;
            return Ok(Some(prim_array.into_array()));
        }

        Ok(Some(primitive.into_array()))
    }
}

/// Specialized filter kernel for primitive bit-packed arrays.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, the provided
/// `U` should be promoted to the unsigned variant for any target bit width.
/// For example, if the array is bit-packed `i16`, this function should be called with `U = u16`.
///
/// This function fully decompresses the array for all but the most selective masks because the
/// FastLanes decompression is so fast and the bookkeepping necessary to decompress individual
/// elements is relatively slow.
///
/// Returns a tuple of (values buffer, validity mask).
fn filter_primitive_without_patches<U: UnsignedPType + BitPacking>(
    array: ArrayView<'_, BitPacked>,
    selection: &Arc<MaskValues>,
) -> VortexResult<(Buffer<U>, Validity)> {
    let values = filter_with_indices(array.data(), selection.indices());
    let validity = array
        .validity()?
        .filter(&Mask::Values(Arc::clone(selection)))?;

    Ok((values.freeze(), validity))
}

fn filter_with_indices<T: NativePType + BitPacking>(
    array: &BitPackedData,
    indices: &[usize],
) -> BufferMut<T> {
    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;
    let mut values = BufferMut::with_capacity(indices.len());

    // Some re-usable memory to store per-chunk indices.
    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; 1024];
    let packed_bytes = array.packed_slice::<T>();

    // Group the indices by the FastLanes chunk they belong to.
    let chunk_size = 128 * bit_width / size_of::<T>();

    chunked_indices(
        indices.iter().copied(),
        offset,
        |chunk_idx, indices_within_chunk| {
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
                    let dst: &mut [T] = std::mem::transmute(dst);
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
        },
    );

    values
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray as _;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    #[test]
    fn take_indices() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = bitpacked.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18])
        );
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = sliced.filter(mask).unwrap();
        assert_arrays_eq!(primitive_result, PrimitiveArray::from_iter([31u8, 33]));
    }

    #[test]
    fn filter_bitpacked() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(4096, (0..1024).collect()))
            .unwrap();
        let filtered_prim = filtered.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            filtered_prim,
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8))
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 9, &mut ctx).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), (0..250).collect()))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied())
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array());

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 9, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array());

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array());
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create signed integer values where some exceed the bit width (causing patches).
        // Values 0-127 fit in 7 bits, but 1000 and 2000 do not.
        let values: Vec<i32> = vec![0, 10, 1000, 20, 30, 2000, 40, 50, 60, 70];
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Filter to include some patched and some non-patched values.
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), vec![0, 2, 5, 9]))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(filtered, PrimitiveArray::from_iter([0i32, 1000, 2000, 70]));
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Use low selectivity (only select 2% of values) to avoid full decompression.
        let indices: Vec<usize> = (0..20).collect();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), indices))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        let expected: Vec<i32> = values[0..20].to_vec();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected));
    }
}
