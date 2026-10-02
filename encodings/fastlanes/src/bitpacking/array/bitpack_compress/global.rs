// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-packing every block of an array at one global bit width.

use fastlanes::BitPacking;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::bit_width_histogram;
use super::bitpacked_from_packed;
use super::ensure_non_negative_integers;
use super::find_best_bit_width;
use super::gather_patches;
use super::pack_blocks;
use super::pack_blocks_unchecked;
use crate::BitPackedArray;
use crate::BitWidths;
use crate::bitpack_decompress;

pub fn bitpack_to_best_bit_width(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    let bit_width_freq = bit_width_histogram(array.as_view(), ctx)?;
    let best_bit_width = find_best_bit_width(array.ptype(), &bit_width_freq)?;
    bitpack_encode(array, best_bit_width, Some(&bit_width_freq), ctx)
}

pub fn bitpack_encode(
    array: &PrimitiveArray,
    bit_width: u8,
    bit_width_freq: Option<&[usize]>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    ensure_non_negative_integers(array, ctx)?;
    let bit_width_freq = match bit_width_freq {
        Some(freq) => freq,
        None => &bit_width_histogram(array.as_view(), ctx)?,
    };

    let num_exceptions = bitpack_decompress::count_exceptions(bit_width, bit_width_freq);

    if bit_width >= array.ptype().bit_width() as u8 {
        // Nothing we can do
        vortex_bail!(
            InvalidArgument: "Cannot pack - specified bit width {bit_width} >= {}",
            array.ptype().bit_width()
        )
    }

    // SAFETY: we check that array only contains non-negative values.
    let packed = unsafe { bitpack_unchecked(array, bit_width) };
    let patches = (num_exceptions > 0)
        .then(|| gather_patches(array, bit_width, num_exceptions, ctx))
        .transpose()?
        .flatten();
    bitpacked_from_packed(array, packed, patches, BitWidths::Global(bit_width))
}

/// Bitpack an array into the specified bit-width without checking statistics.
///
/// # Safety
///
/// It is the caller's responsibility to ensure that all values in the array can lossless pack
/// into the specified bit-width.
///
/// Failure to do so will result in data loss.
pub unsafe fn bitpack_encode_unchecked(
    array: PrimitiveArray,
    bit_width: u8,
) -> VortexResult<BitPackedArray> {
    // SAFETY: non-negativity of input checked by caller.
    let packed = unsafe { bitpack_unchecked(&array, bit_width) };
    bitpacked_from_packed(&array, packed, None, BitWidths::Global(bit_width))
}

/// Bitpack a [PrimitiveArray] to the given width.
///
/// On success, returns a [Buffer] containing the packed data.
///
/// # Safety
///
/// Internally this function will promote the provided array to its unsigned equivalent. This will
/// violate ordering guarantees if the array contains any negative values.
///
/// It is the caller's responsibility to ensure that `parray` is non-negative before calling
/// this function.
pub unsafe fn bitpack_unchecked(parray: &PrimitiveArray, bit_width: u8) -> ByteBuffer {
    // SAFETY: the caller ensures that `parray` is non-negative.
    unsafe { pack_blocks_unchecked(parray, &|_| bit_width) }
}

/// Bitpack a slice of primitives down to the given width.
///
/// See `bitpack` for more caller information.
pub fn bitpack_primitive<T: NativePType + BitPacking>(array: &[T], bit_width: u8) -> Buffer<T> {
    pack_blocks(array, &|_| bit_width)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::bitpack_compress::test_harness::make_array;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn null_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let valid_values = (0..24).map(|v| v < 1 << 4).collect::<Vec<_>>();
        let values = PrimitiveArray::new(
            (0u32..24).collect::<Buffer<_>>(),
            Validity::from_iter(valid_values),
        );
        assert!(values.ptype().is_unsigned_int());
        let compressed = BitPackedData::encode(&values.into_array(), 4, &mut ctx).unwrap();
        assert!(compressed.patches().is_none());
        assert_eq!(
            (0..(1 << 4)).collect::<Vec<_>>(),
            compressed
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(compressed.as_ref().len(), &mut ctx)
                .unwrap()
                .to_bit_buffer()
                .set_indices()
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn compress_signed_fails() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i64> = (-500..500).collect();
        let array = PrimitiveArray::new(values, Validity::AllValid);
        assert!(array.ptype().is_signed_int());

        let err = BitPackedData::encode(&array.into_array(), 1024u32.ilog2() as u8, &mut ctx)
            .unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }

    #[test]
    fn canonicalize_chunked_of_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(0);

        let chunks = (0..10)
            .map(|_| make_array(&mut rng, 100, 0.25, 0.25, &mut ctx).unwrap())
            .collect::<Vec<_>>();
        let chunked = ChunkedArray::from_iter(chunks).into_array();

        let into_ca = chunked.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let mut primitive_builder = PrimitiveBuilder::<i32>::with_capacity_in(
            chunked.dtype().nullability(),
            10 * 100,
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        chunked.append_to_builder(&mut primitive_builder, &mut ctx)?;
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into, &mut ctx);

        let mut primitive_builder = PrimitiveBuilder::<i32>::with_capacity_in(
            chunked.dtype().nullability(),
            10 * 100,
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        chunked.append_to_builder(&mut primitive_builder, &mut ctx)?;
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_chunk_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 3000, 3100];
        let mut values = vec![0u32; 4096usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None, &mut ctx)?;

        let patches = bitpacked
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        let chunk_offsets = patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        // chunk 0 (0-1023): patches at 100, 200 -> starts at patch index 0
        // chunk 1 (1024-2047): no patches -> points to patch index 2
        // chunk 2 (2048-3071): patch at 3000 -> starts at patch index 2
        // chunk 3 (3072-4095): patch at 3100 -> starts at patch index 3
        assert_arrays_eq!(
            chunk_offsets,
            PrimitiveArray::from_iter([0u64, 2, 2, 3]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_chunk_offsets_no_patches_in_middle() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 2500];
        let mut values = vec![0u32; 3072usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None, &mut ctx)?;

        let patches = bitpacked
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        let chunk_offsets = patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(
            chunk_offsets,
            PrimitiveArray::from_iter([0u64, 2, 2]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_chunk_offsets_trailing_empty_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 1500];
        let mut values = vec![0u32; 5120usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None, &mut ctx)?;

        let patches = bitpacked
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        let chunk_offsets = patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        // chunk 0 (0-1023): patches at 100, 200 -> starts at patch index 0
        // chunk 1 (1024-2047): patch at 1500 -> starts at patch index 2
        // chunk 2 (2048-3071): no patches -> points to patch index 3
        // chunk 3 (3072-4095): no patches -> points to patch index 3 (remaining chunks filled)
        // chunk 4 (4096-5119): no patches -> points to patch index 3 (remaining chunks filled)
        assert_arrays_eq!(
            chunk_offsets,
            PrimitiveArray::from_iter([0u64, 2, 3, 3, 3]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_chunk_offsets_single_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200];
        let mut values = vec![0u32; 500usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None, &mut ctx)?;

        let patches = bitpacked
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        let chunk_offsets = patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        // Single chunk starting at patch index 0.
        assert_arrays_eq!(chunk_offsets, PrimitiveArray::from_iter([0u64]), &mut ctx);
        Ok(())
    }
}
