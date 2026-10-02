// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-packing every 1024-value block of an array at its own bit width.

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::ensure_non_negative_integers;
use super::find_best_bit_width;
use super::global::bitpack_blocks;
use super::global::gather_patches_with;
use crate::BitPacked;
use crate::BitPackedArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_decompress;

/// Bit-pack `array`, choosing the bit width of every 1024-value block.
///
/// Each block uses the width that minimizes its packed size plus the cost of its exceptions. The
/// block offsets are always materialized, even when every block chooses the same width.
///
/// # Errors
///
/// Returns an error if `array` is not an integer array or contains negative values.
pub fn bitpack_to_best_bit_widths(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    ensure_non_negative_integers(array, ctx)?;
    let validity_mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let (bit_widths, num_exceptions) = match_each_integer_ptype!(array.ptype(), |T| {
        block_bit_widths(array.as_slice::<T>(), &validity_mask)?
    });
    bitpack_encode_blocked(array, &bit_widths, Some(num_exceptions), ctx)
}

/// Bit-pack each 1024-value block of `array` at its width in `bit_widths`.
///
/// Valid values wider than their block's width become patches. `num_exceptions` is the number of
/// such values when the caller knows it: it sizes the patches, and zero skips gathering them.
///
/// # Errors
///
/// Returns an error if `array` is not an integer array or contains negative values, or if
/// `bit_widths` does not hold one width per block of at most the array's bit width.
pub fn bitpack_encode_blocked(
    array: &PrimitiveArray,
    bit_widths: &[u8],
    num_exceptions: Option<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    ensure_non_negative_integers(array, ctx)?;
    let num_blocks = array.len().div_ceil(FL_CHUNK_SIZE);
    vortex_ensure!(
        bit_widths.len() == num_blocks,
        InvalidArgument: "Expected {num_blocks} bit widths, got {}",
        bit_widths.len()
    );
    let max_bit_width = array.ptype().bit_width();
    vortex_ensure!(
        bit_widths
            .iter()
            .all(|&bit_width| usize::from(bit_width) <= max_bit_width),
        InvalidArgument: "Bit widths must be at most {max_bit_width} for {}",
        array.ptype()
    );

    // SAFETY: we check that array only contains non-negative values.
    let packed = unsafe { bitpack_blocked_unchecked(array, bit_widths) };
    let patches = if num_exceptions == Some(0) {
        None
    } else {
        gather_patches_with(
            array,
            |block| bit_widths[block],
            num_exceptions.unwrap_or(0),
            ctx,
        )?
    };

    let bitpacked = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        patches,
        block_offsets_from_widths(bit_widths),
        array.len(),
        0,
    )?;
    bitpacked.statistics().inherit_from(array.statistics());
    Ok(bitpacked)
}

/// Bitpack each 1024-value block of `array` at its width in `bit_widths`.
///
/// # Safety
///
/// This promotes `array` to its unsigned equivalent, like [`bitpack_unchecked`], so the caller
/// must ensure that it holds no negative values.
///
/// [`bitpack_unchecked`]: super::bitpack_unchecked
unsafe fn bitpack_blocked_unchecked(array: &PrimitiveArray, bit_widths: &[u8]) -> ByteBuffer {
    let array = array.reinterpret_cast(array.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(array.ptype(), |T| {
        let capacity = bit_widths
            .iter()
            .map(|&bit_width| 128 * usize::from(bit_width) / size_of::<T>())
            .sum();
        bitpack_blocks(array.as_slice::<T>(), capacity, |block| {
            usize::from(bit_widths[block])
        })
        .into_byte_buffer()
    })
}

/// Byte boundaries of blocks packed at `widths`, in the narrowest unsigned type that holds them.
fn block_offsets_from_widths(widths: &[u8]) -> ArrayRef {
    let end: u64 = widths.iter().map(|&width| 128 * u64::from(width)).sum();
    let ptype = PType::min_unsigned_ptype_for_value(end);
    match_each_unsigned_integer_ptype!(ptype, |T| {
        let mut offsets = BufferMut::<T>::with_capacity(widths.len() + 1);
        let mut offset = 0u64;
        offsets.push(offset.as_());
        for &width in widths {
            offset += 128 * u64::from(width);
            offsets.push(offset.as_());
        }
        offsets.into_array()
    })
}

/// The width minimizing each 1024-value block's packed size plus the cost of its exceptions, and
/// the total number of exceptions those widths leave.
fn block_bit_widths<T: NativePType + PrimInt>(
    values: &[T],
    validity_mask: &Mask,
) -> VortexResult<(Vec<u8>, usize)> {
    let mut histogram = vec![0usize; size_of::<T>() * 8 + 1];
    let mut widths = Vec::with_capacity(values.len().div_ceil(FL_CHUNK_SIZE));
    let mut num_exceptions = 0;
    for (block_idx, block) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        // The zero padding of a partial block and null values need no bits, so counting them as
        // zero-width charges every block for its full packed size.
        histogram.fill(0);
        histogram[0] = FL_CHUNK_SIZE - block.len();
        let start = block_idx * FL_CHUNK_SIZE;
        let block_validity = validity_mask.slice(start..start + block.len());
        add_bit_widths(&mut histogram, block, block_validity.bit_buffer());

        let width = find_best_bit_width(T::PTYPE, &histogram)?;
        num_exceptions += bitpack_decompress::count_exceptions(width, &histogram);
        widths.push(width);
    }

    Ok((widths, num_exceptions))
}

/// Count the bit width of each of `values` into `histogram`, counting null values as zero-width.
fn add_bit_widths<T: NativePType + PrimInt>(
    histogram: &mut [usize],
    values: &[T],
    validity: AllOr<&BitBuffer>,
) {
    let bit_width = |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);
    match validity {
        AllOr::All => {
            for v in values {
                histogram[bit_width(*v)] += 1;
            }
        }
        AllOr::None => histogram[0] += values.len(),
        AllOr::Some(buffer) => {
            for (is_valid, v) in buffer.iter().zip_eq(values) {
                histogram[if is_valid { bit_width(*v) } else { 0 }] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Primitive;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::BitWidthsView;
    use crate::bitpack_compress::bitpack_primitive;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// The materialized block offsets of a blocked array.
    fn block_offsets(
        array: &BitPackedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let BitWidthsView::Blocked(offsets) = array.bit_widths() else {
            vortex_bail!("expected block offsets");
        };
        offsets.clone().execute::<PrimitiveArray>(ctx)
    }

    /// The bit width of each block, from the distance between its boundaries.
    fn block_widths(array: &BitPackedArray, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u64>> {
        let offsets = block_offsets(array, ctx)?;
        Ok(match_each_unsigned_integer_ptype!(offsets.ptype(), |T| {
            offsets
                .as_slice::<T>()
                .windows(2)
                .map(|pair| {
                    (AsPrimitive::<u64>::as_(pair[1]) - AsPrimitive::<u64>::as_(pair[0])) / 128
                })
                .collect()
        }))
    }

    /// Every block packed on its own at its width, one after another.
    fn pack_blocks(values: &[u32], widths: &[u8]) -> Vec<u32> {
        values
            .chunks(1024)
            .zip_eq(widths)
            .flat_map(|(block, &width)| bitpack_primitive(block, width).to_vec())
            .collect()
    }

    #[test]
    fn best_bit_widths_choose_each_block_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..1024)
            .map(|i| i % 8)
            .chain(std::iter::repeat_n(0, 1024))
            .chain((0..1024).map(|i| (1 << 19) + i))
            .chain((0..100).map(|i| i % 32))
            .collect();
        let array = bitpack_to_best_bit_widths(
            &PrimitiveArray::from_iter(values.iter().copied()),
            &mut ctx,
        )?;

        assert_eq!(block_widths(&array, &mut ctx)?, [3, 0, 20, 5]);
        assert_eq!(block_offsets(&array, &mut ctx)?.ptype(), PType::U16);
        assert_eq!(
            array.packed_slice::<u32>(),
            pack_blocks(&values, &[3, 0, 20, 5]).as_slice()
        );
        assert!(array.patches().is_none());
        Ok(())
    }

    #[test]
    fn encode_blocked_patches_use_each_block_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values: Vec<i32> = (0..1024)
            .map(|i| i % 8)
            .chain(std::iter::repeat_n(1 << 19, 1024))
            .collect();
        // 2^19 is an exception in the 3-bit block but fits the 20-bit block.
        values[10] = 1 << 19;
        values[20] = (1 << 20) + 1;
        values[1029] = (1 << 20) + 1;
        let array = bitpack_encode_blocked(
            &PrimitiveArray::from_iter(values.iter().copied()),
            &[3, 20],
            None,
            &mut ctx,
        )?;

        assert_eq!(block_widths(&array, &mut ctx)?, [3, 20]);
        let unsigned: Vec<u32> = values.iter().map(|v| v.cast_unsigned()).collect();
        assert_eq!(
            array.packed_slice::<u32>(),
            pack_blocks(&unsigned, &[3, 20]).as_slice()
        );

        let patches = array
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        assert_arrays_eq!(
            patches
                .indices()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?,
            PrimitiveArray::from_iter([10u16, 20, 1029]),
            &mut ctx
        );
        assert_arrays_eq!(
            patches
                .values()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?,
            PrimitiveArray::from_iter([1i32 << 19, (1 << 20) + 1, (1 << 20) + 1]),
            &mut ctx
        );
        let chunk_offsets = patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            chunk_offsets,
            PrimitiveArray::from_iter([0u64, 2]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn best_bit_widths_charge_partial_blocks_for_padding() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Packing these values would take a full 2688-byte block; patching them takes 80 bytes.
        let array = bitpack_to_best_bit_widths(
            &PrimitiveArray::from_iter(vec![(1u32 << 20) + 1; 10]),
            &mut ctx,
        )?;

        assert_eq!(block_widths(&array, &mut ctx)?, [0]);
        assert_eq!(array.packed().len(), 0);
        let patches = array
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        assert_eq!(patches.num_patches(), 10);
        Ok(())
    }

    #[test]
    fn best_bit_widths_materialize_uniform_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = bitpack_to_best_bit_widths(
            &PrimitiveArray::from_iter((0..3000u32).map(|i| i % 128)),
            &mut ctx,
        )?;

        let BitWidthsView::Blocked(offsets) = array.bit_widths() else {
            vortex_bail!("expected block offsets");
        };
        assert!(offsets.is::<Primitive>());
        assert_arrays_eq!(
            block_offsets(&array, &mut ctx)?,
            PrimitiveArray::from_iter([0u16, 896, 1792, 2688]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn best_bit_widths_ignore_null_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Null slots hold values wider than their block, which are neither counted nor patched.
        let values = PrimitiveArray::new(
            (0..2048u32)
                .map(|i| if i % 10 == 0 { u32::MAX } else { i % 16 })
                .collect::<Buffer<_>>(),
            Validity::from_iter((0..2048).map(|i| i % 10 != 0)),
        );
        let array = bitpack_to_best_bit_widths(&values, &mut ctx)?;

        assert_eq!(block_widths(&array, &mut ctx)?, [4, 4]);
        assert!(array.patches().is_none());
        Ok(())
    }

    #[test]
    fn best_bit_widths_of_empty_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            bitpack_to_best_bit_widths(&PrimitiveArray::from_iter(Vec::<u32>::new()), &mut ctx)?;

        assert_arrays_eq!(
            block_offsets(&array, &mut ctx)?,
            PrimitiveArray::from_iter([0u8]),
            &mut ctx
        );
        assert_eq!(array.packed().len(), 0);
        Ok(())
    }

    #[rstest]
    #[case::negative(PrimitiveArray::from_iter(-5i64..5))]
    #[case::float(PrimitiveArray::from_iter([1.0f32, 2.0]))]
    fn encode_blocked_rejects_invalid_values(#[case] array: PrimitiveArray) {
        let mut ctx = SESSION.create_execution_ctx();
        assert!(matches!(
            bitpack_to_best_bit_widths(&array, &mut ctx).unwrap_err(),
            VortexError::InvalidArgument(_, _)
        ));
        assert!(matches!(
            BitPackedData::encode_blocked(&array.into_array(), &[1], &mut ctx).unwrap_err(),
            VortexError::InvalidArgument(_, _)
        ));
    }

    #[rstest]
    #[case::too_few(vec![3])]
    #[case::too_many(vec![3, 3, 3])]
    #[case::wider_than_type(vec![3, 33])]
    fn encode_blocked_rejects_invalid_bit_widths(#[case] bit_widths: Vec<u8>) {
        let array = PrimitiveArray::from_iter((0..2048u32).map(|i| i % 8));
        let err = BitPackedData::encode_blocked(
            &array.into_array(),
            &bit_widths,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }

    #[test]
    fn encode_blocked_accepts_native_bit_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..1024).map(|i| u32::MAX - i).collect();
        let array = bitpack_encode_blocked(
            &PrimitiveArray::from_iter(values.iter().copied()),
            &[32],
            Some(0),
            &mut ctx,
        )?;
        assert_eq!(
            array.packed_slice::<u32>(),
            pack_blocks(&values, &[32]).as_slice()
        );
        assert!(array.patches().is_none());
        Ok(())
    }
}
