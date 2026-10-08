// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::ChunkDecoder;
use vortex_array::arrays::filter::FILTER_CHUNK_LEN;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::filter::filter_chunked;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::BitWidthsView;

/// Kernel to execute filtering directly on a bit-packed array.
///
/// [`filter_chunked`] unpacks only the FastLanes chunks that hold selected values, and compacts
/// each chunk while it is in cache, rather than unpack the whole array and then filter it.
impl FilterKernel for BitPacked {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        let values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => {
                return Ok(None);
            }
            Mask::Values(values) => values,
        };

        // FastLanes only unpacks unsigned types, so filter as unsigned and reinterpret the
        // resulting buffer with the array's (possibly signed) ptype.
        let ptype = array.dtype().as_ptype();
        let validity = array.validity()?.filter(mask)?;
        let buffer = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            filter_values::<U>(array.data(), bit_width, values).into_byte_buffer()
        });
        let primitive = PrimitiveArray::from_byte_buffer(buffer, ptype, validity);

        let patches = array
            .patches()
            .map(|patches| patches.filter(mask, ctx))
            .transpose()?
            .flatten();

        Ok(Some(match patches {
            Some(patches) => primitive.patch(&patches, ctx)?.into_array(),
            None => primitive.into_array(),
        }))
    }
}

/// Unpacks the values of `array` selected by `mask`, ignoring patches and validity.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, `T` must
/// be the unsigned variant of the array's ptype.
fn filter_values<T: NativePType + BitPacking>(
    array: &BitPackedData,
    bit_width: u8,
    mask: &MaskValues,
) -> Buffer<T> {
    let bit_width = bit_width as usize;
    let chunks = PackedChunks {
        packed: array.packed_slice::<T>(),
        bit_width,
        packed_chunk_len: 128 * bit_width / size_of::<T>(),
    };
    filter_chunked(&chunks, array.offset() as usize, mask)
}

/// The packed FastLanes chunks of a bit-packed array.
struct PackedChunks<'a, T> {
    packed: &'a [T],
    bit_width: usize,
    /// Number of `T` words that hold each packed chunk.
    packed_chunk_len: usize,
}

impl<T> PackedChunks<'_, T> {
    fn chunk(&self, chunk_idx: usize) -> &[T] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: the FastLanes unpack kernels initialize every value of `dst`.
unsafe impl<T: BitPacking> ChunkDecoder<T> for PackedChunks<'_, T> {
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        // SAFETY: `MaybeUninit<T>` has the same layout as `T`, the unpack only writes to `dst`,
        // and the packed chunk holds `FILTER_CHUNK_LEN` values of `bit_width` bits.
        unsafe {
            let dst = &mut *(dst as *mut [MaybeUninit<T>] as *mut [T]);
            BitPacking::unchecked_unpack(self.bit_width, self.chunk(chunk_idx), dst);
        }
    }

    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        debug_assert!(indices.iter().all(|&index| index < FILTER_CHUNK_LEN));
        // SAFETY: every index is within the chunk, and `dst` holds one value for each index.
        unsafe {
            BitPacking::unchecked_unpack_indices(
                self.bit_width,
                self.chunk(chunk_idx),
                indices,
                dst,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use fastlanes::BitPacking;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::filter_values;
    use crate::BitPacked;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn take_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = bitpacked.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18]),
            &mut ctx
        );
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = sliced.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([31u8, 33]),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked() {
        let mut ctx = SESSION.create_execution_ctx();
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let filtered = bitpacked.filter(Mask::from_indices(4096, 0..1024)).unwrap();
        let filtered_prim = filtered.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            filtered_prim,
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8)),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 9, &mut ctx).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), 0..250))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied()),
            &mut ctx
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        let mut ctx = SESSION.create_execution_ctx();
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 9, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
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

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([0i32, 1000, 2000, 70]),
            &mut ctx
        );
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        let mut ctx = SESSION.create_execution_ctx();
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
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected), &mut ctx);
    }

    /// Selection patterns that exercise every chunk strategy of the kernel.
    #[derive(Clone, Copy, Debug)]
    enum Pattern {
        /// Each value is selected independently with the given probability.
        Random(f64),
        /// Runs of 12 values every 64 values.
        Runs,
        /// Half of the values in the first 400 of every 1024 positions, which are too dense
        /// within a chunk to unpack individually.
        Clustered,
        /// Everything except a single value, so most chunks are fully selected.
        AllButOne,
    }

    fn selection(pattern: Pattern, len: usize) -> BitBuffer {
        let mut rng = StdRng::seed_from_u64(42);
        BitBuffer::from_iter((0..len).map(|i| match pattern {
            Pattern::Random(density) => rng.random_bool(density),
            Pattern::Runs => i % 64 < 12,
            Pattern::Clustered => i % 1024 < 400 && rng.random_bool(0.5),
            Pattern::AllButOne => i != len / 2,
        }))
    }

    /// A bit-packed array of `len + 2000` values with some patches, sliced to `range`.
    fn sliced_bitpacked(len: usize, range: Range<usize>) -> VortexResult<(ArrayRef, ArrayRef)> {
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter((0..len as u32).map(|i| {
                (i % 97 != 0).then_some(if i % 501 == 0 { 100_000 + i } else { i % 1000 })
            }));
        let bitpacked = BitPackedData::encode(&values.clone().into_array(), 10, &mut ctx)?;
        assert!(bitpacked.patches().is_some());
        // Slicing patches needs execution, so slice with the kernel rather than lazily.
        let sliced =
            <BitPacked as SliceKernel>::slice(bitpacked.as_view(), range.clone(), &mut ctx)?
                .unwrap();
        Ok((values.into_array().slice(range)?, sliced))
    }

    #[rstest]
    fn filter_matches_canonical(
        #[values(
            Pattern::Random(0.0005),
            Pattern::Random(0.01),
            Pattern::Random(0.05),
            Pattern::Random(0.15),
            Pattern::Random(0.5),
            Pattern::Runs,
            Pattern::Clustered,
            Pattern::AllButOne
        )]
        pattern: Pattern,
        #[values(0..5000, 3..5000, 1000..4099, 1024..3072)] range: Range<usize>,
        #[values(false, true)] cached_slices: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (values, bitpacked) = sliced_bitpacked(7000, range)?;
        assert!(bitpacked.is::<BitPacked>());

        let bits = selection(pattern, values.len());
        let mask = if cached_slices {
            Mask::from_slices(bits.len(), bits.set_slices().collect())
        } else {
            Mask::from_buffer(bits)
        };

        let expected = values
            .filter(mask.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let actual = bitpacked
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Checks every value width, since the number of selected values for which a chunk unpacks
    /// only those values depends on the width.
    #[rstest]
    fn filter_values_matches_expected_for_every_width(
        #[values(
            Pattern::Random(0.01),
            Pattern::Random(0.03),
            Pattern::Random(0.08),
            Pattern::Random(0.15),
            Pattern::Runs,
            Pattern::Clustered
        )]
        pattern: Pattern,
    ) -> VortexResult<()> {
        check_filter_values::<u8>(pattern)?;
        check_filter_values::<u16>(pattern)?;
        check_filter_values::<u32>(pattern)?;
        check_filter_values::<u64>(pattern)
    }

    fn check_filter_values<T: NativePType + BitPacking + From<u8>>(
        pattern: Pattern,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let range = 3..5000;

        let unpacked =
            PrimitiveArray::from_iter((0..5000u32).map(|i| <T as From<u8>>::from((i % 100) as u8)));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx)?
            .into_array()
            .slice(range.clone())?;
        let bitpacked = bitpacked.as_::<BitPacked>();

        let bits = selection(pattern, range.len());
        let expected: Vec<T> = range
            .zip(bits.iter())
            .filter_map(|(i, selected)| selected.then_some(<T as From<u8>>::from((i % 100) as u8)))
            .collect();

        let Mask::Values(mask) = Mask::from_buffer(bits) else {
            unreachable!("patterns select some but not all values")
        };
        assert_eq!(
            filter_values::<T>(bitpacked.data(), 7, &mask).as_slice(),
            expected
        );
        Ok(())
    }
}
