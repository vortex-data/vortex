// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding bit-packed arrays whose blocks each have their own bit width.

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::global::apply_patches_to_uninit_range;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::block_range;
use crate::bitpacking::array::validate_primitive_offsets;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Unpacks a bit-packed array with block `offsets` into a primitive array.
pub fn unpack_array_blocked(
    array: ArrayView<'_, BitPacked>,
    offsets: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
        unpack_primitive_array_blocked::<P>(array, offsets, ctx)
    })
}

pub fn unpack_primitive_array_blocked<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    offsets: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mut builder = PrimitiveBuilder::with_capacity_in(
        array.dtype().nullability(),
        array.len(),
        ctx.allocator(),
    );
    unpack_into_primitive_builder_blocked::<T>(array, offsets, &mut builder, ctx)?;
    assert_eq!(builder.len(), array.len());
    Ok(builder.finish_into_primitive())
}

/// Unpack a bit-packed array with block `offsets` directly into a same-typed `PrimitiveBuilder`.
///
/// The offsets are executed and validated before the builder is touched. Full blocks are unpacked
/// straight into the output; a partial first or last block is unpacked into a scratch block.
pub(crate) fn unpack_into_primitive_builder_blocked<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    offsets: &ArrayRef,
    builder: &mut PrimitiveBuilder<T>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if array.is_empty() {
        return Ok(());
    }
    assert_eq!(
        T::PTYPE,
        array.dtype().as_ptype(),
        "Requested type doesn't match the array ptype"
    );

    // Decoding one offset type compiles the block loop once per value type.
    let offsets = offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)?;
    let num_blocks = (array.len() + usize::from(array.offset())).div_ceil(FL_CHUNK_SIZE);
    vortex_ensure!(
        offsets.len() == num_blocks + 1,
        "Expected {} block boundaries, got {}",
        num_blocks + 1,
        offsets.len()
    );
    let offsets = Buffer::<u64>::from_byte_buffer(offsets.buffer_handle().try_to_host_sync()?);
    validate_primitive_offsets(
        &offsets,
        array.dtype().as_ptype().bit_width() as u64,
        array.packed().len(),
    )?;

    let len = array.len();
    let validity = array.validity()?.execute_mask(len, ctx)?;
    let mut uninit_range = builder.uninit_range(len);

    // SAFETY: We initialize all `len` values below via `decode_blocks` and the patch loop.
    unsafe {
        uninit_range.append_mask(&validity);
    }

    // SAFETY: `decode_blocks` writes a value to every slot in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, len) };

    decode_blocks(array, &offsets, uninit_slice);

    if let Some(patches) = array.patches() {
        apply_patches_to_uninit_range(&mut uninit_range, &patches, ctx, |v: T| v)?;
    }

    // SAFETY: A correct validity mask of `len` values was set via `append_mask`, and the same
    // number of values was initialized via `decode_blocks` (and overwritten by patches).
    unsafe {
        uninit_range.finish();
    }
    Ok(())
}

/// Decode the blocks between validated `offsets` into `output`.
fn decode_blocks<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    offsets: &[u64],
    output: &mut [MaybeUninit<T>],
) {
    let packed = array.packed_slice::<T::Physical>();
    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    let base = offsets[0];
    let mut skip = usize::from(array.offset());
    let mut written = 0;
    for pair in offsets.windows(2) {
        // Validation bounds these differences by the packed buffer's usize length.
        let start = (pair[0] - base) as usize;
        let end = (pair[1] - base) as usize;
        let bit_width = (end - start) / 128;
        let block = &packed[start / size_of::<T>()..end / size_of::<T>()];
        let len = (FL_CHUNK_SIZE - skip).min(output.len() - written);
        let dst = &mut output[written..][..len];
        if len == FL_CHUNK_SIZE {
            // SAFETY: The boundaries have been validated against the packed length and physical
            // type, and `dst` holds exactly one block. `T` and its physical type have the same
            // layout.
            unsafe {
                let dst: &mut [T::Physical] = mem::transmute(dst);
                BitPacking::unchecked_unpack(bit_width, block, dst);
            }
        } else {
            // SAFETY: As above, with the scratch block as the destination.
            unsafe {
                let unpacked: &mut [T::Physical] = mem::transmute(&mut scratch[..]);
                BitPacking::unchecked_unpack(bit_width, block, unpacked);
            }
            dst.copy_from_slice(&scratch[skip..][..len]);
        }
        written += len;
        skip = 0;
    }
    debug_assert_eq!(written, output.len());
}

/// Decode a single value of a bit-packed array with block `offsets`, without applying patches.
///
/// Only the boundaries of the value's block are read and validated.
pub fn unpack_single_blocked(
    array: ArrayView<'_, BitPacked>,
    offsets: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let index_in_encoded = index + array.offset() as usize;
    let block = index_in_encoded / FL_CHUNK_SIZE;
    vortex_ensure!(
        block + 1 < offsets.len(),
        "BitPacked index {index} has no block boundaries"
    );
    let (base, start, end) = if let Some(primitive) = offsets.as_opt::<Primitive>()
        && primitive.buffer_handle().is_on_host()
    {
        match_each_unsigned_integer_ptype!(primitive.ptype(), |I| {
            let offsets = primitive.as_slice::<I>();
            (
                AsPrimitive::<u64>::as_(offsets[0]),
                AsPrimitive::<u64>::as_(offsets[block]),
                AsPrimitive::<u64>::as_(offsets[block + 1]),
            )
        })
    } else {
        let base = u64::try_from(&offsets.execute_scalar(0, ctx)?)?;
        let start = if block == 0 {
            base
        } else {
            u64::try_from(&offsets.execute_scalar(block, ctx)?)?
        };
        let end = u64::try_from(&offsets.execute_scalar(block + 1, ctx)?)?;
        (base, start, end)
    };
    let range = block_range(
        base,
        start,
        end,
        array.dtype().as_ptype().bit_width() as u64,
        array.packed().len(),
    )?;
    let bit_width = range.len() / 128;
    match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
        let packed = array.packed_slice::<<P as PhysicalPType>::Physical>();
        let packed = &packed[range.start / size_of::<P>()..range.end / size_of::<P>()];
        // SAFETY: The boundaries validate the block's length and width against the physical type.
        // The index is within this block, and signed types use the same physical bits.
        let value: P = unsafe {
            BitPacking::unchecked_unpack_single(bit_width, packed, index_in_encoded % FL_CHUNK_SIZE)
        }
        .as_();
        Ok(Scalar::primitive(value, array.dtype().nullability()))
    })
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::PhysicalPType;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::match_each_unsigned_integer_ptype;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedArrayExt;
    use crate::BitWidthsView;
    use crate::FL_CHUNK_SIZE;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_blocked_to_best_bit_widths;
    use crate::bitpack_compress::bitpack_encode_blocked;
    use crate::bitpack_compress::bitpack_primitive;
    use crate::test::SESSION;

    fn variable(
        ptype: PType,
        offset: u16,
        len: usize,
        encoded_offsets: bool,
    ) -> VortexResult<(BitPackedArray, PrimitiveArray)> {
        match_each_integer_ptype!(ptype, |T| {
            type U = <T as PhysicalPType>::Physical;
            let widths = [3, 0, ptype.bit_width() as u8, 5];
            let num_blocks = (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE);
            let mut packed = BufferMut::<U>::with_capacity(num_blocks * FL_CHUNK_SIZE);
            let mut values: Vec<T> = Vec::new();
            let mut boundaries = vec![128u64];
            for &width in &widths[..num_blocks] {
                let mask = 1u64
                    .checked_shl(u32::from(width))
                    .map_or(u64::MAX, |bit| bit - 1);
                let block: Vec<U> = (0..FL_CHUNK_SIZE)
                    .map(|i| AsPrimitive::<U>::as_(u64::MAX.wrapping_sub(i as u64) & mask))
                    .collect();
                packed.extend_from_slice(&bitpack_primitive(&block, width));
                values.extend(block.iter().map(|&value| AsPrimitive::<T>::as_(value)));
                boundaries.push(128 + (packed.len() * size_of::<U>()) as u64);
            }
            let offsets = PrimitiveArray::from_iter(boundaries).into_array();
            let offsets = if encoded_offsets {
                FoR::try_new(offsets, 0u64.into())?.into_array()
            } else {
                offsets
            };
            let expected =
                PrimitiveArray::from_iter(values[usize::from(offset)..][..len].iter().copied());
            let array = BitPacked::try_new_with_block_offsets(
                BufferHandle::new_host(packed.freeze().into_byte_buffer()),
                ptype,
                Validity::NonNullable,
                None,
                offsets,
                len,
                offset,
            )?;
            Ok((array, expected))
        })
    }

    #[rstest]
    #[case::full_blocks(0, 4096)]
    #[case::partial_first_and_last(17, 4000)]
    #[case::one_value_header(1023, 2050)]
    #[case::single_full_block(0, 1024)]
    #[case::single_partial_block(17, 100)]
    #[case::one_value(0, 1)]
    #[case::empty(0, 0)]
    fn decode_variable_widths(
        #[values(
            PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
            PType::I64
        )]
        ptype: PType,
        #[case] offset: u16,
        #[case] len: usize,
        #[values(false, true)] encoded_offsets: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (array, expected) = variable(ptype, offset, len, encoded_offsets)?;
        let decoded = array
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        for index in [
            0,
            1,
            1006,
            1007,
            1008,
            1023,
            1024,
            2031,
            2048,
            len.saturating_sub(1),
        ] {
            if index < len {
                assert_eq!(
                    array.execute_scalar(index, &mut ctx)?,
                    expected.execute_scalar(index, &mut ctx)?
                );
            }
        }
        Ok(())
    }

    #[rstest]
    fn decode_unsigned_offsets(
        #[values(PType::U8, PType::U16, PType::U32, PType::U64)] ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let offsets = match_each_unsigned_integer_ptype!(ptype, |T| {
            PrimitiveArray::from_iter([127u8, 255, 255].map(T::from)).into_array()
        });
        let array = BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(ByteBuffer::zeroed(128)),
            PType::U32,
            Validity::NonNullable,
            None,
            offsets,
            2048,
            0,
        )?;
        assert_arrays_eq!(array, PrimitiveArray::from_iter([0u32; 2048]), &mut ctx);
        assert_eq!(array.execute_scalar(0, &mut ctx)?, 0u32.into());
        assert_eq!(array.execute_scalar(1024, &mut ctx)?, 0u32.into());
        Ok(())
    }

    #[test]
    fn decode_zero_width_blocks_with_constant_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(ByteBuffer::empty()),
            PType::U32,
            Validity::NonNullable,
            None,
            ConstantArray::new(u64::MAX, 3).into_array(),
            2048,
            0,
        )?;
        assert_arrays_eq!(array, PrimitiveArray::from_iter([0u32; 2048]), &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_with_nulls_and_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (array, expected) = variable(PType::U32, 17, 3100, true)?;
        let mut values: Vec<_> = expected
            .as_slice::<u32>()
            .iter()
            .copied()
            .map(Some)
            .collect();
        let indices = [0u32, 1007, 1008, 2031, 2032, 3099];
        for &index in &indices {
            values[index as usize] = Some(999);
        }
        values[1008] = None;
        values[2048] = None;
        let expected = PrimitiveArray::from_option_iter(values.iter().copied());
        let patches = Patches::new(
            array.len(),
            113,
            PrimitiveArray::from_iter(indices.map(|i| i + 113)).into_array(),
            PrimitiveArray::new(buffer![999u32; 6], Validity::AllValid).into_array(),
            None,
        )?;
        let BitWidthsView::Blocked(block_offsets) = array.bit_widths() else {
            vortex_bail!("expected block offsets");
        };
        let array = BitPacked::try_new_with_block_offsets(
            array.packed().clone(),
            PType::U32,
            expected.validity()?,
            Some(patches),
            block_offsets.clone(),
            array.len(),
            array.offset(),
        )?;
        let mut builder = PrimitiveBuilder::<u32>::with_capacity_in(
            Nullability::Nullable,
            array.len() + 2,
            ctx.allocator(),
        );
        builder.append_null();
        array.append_to_builder(&mut builder, &mut ctx)?;
        builder.append_value(7);
        let appended =
            PrimitiveArray::from_option_iter([None].into_iter().chain(values).chain([Some(7)]));
        assert_arrays_eq!(builder.finish_into_primitive(), appended, &mut ctx);
        for index in [0, 1007, 1008, 2031, 2032, 2048, 3099] {
            assert_eq!(
                array.execute_scalar(index, &mut ctx)?,
                expected.execute_scalar(index, &mut ctx)?
            );
        }
        Ok(())
    }

    #[test]
    fn compute_falls_back_to_variable_width_decode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (array, expected) = variable(PType::U32, 17, 4000, false)?;
        let array = array.into_array();
        let expected = expected.into_array();
        assert_arrays_eq!(
            array.slice(1000..2200)?,
            expected.slice(1000..2200)?,
            &mut ctx
        );
        let indices = buffer![3999u32, 0, 1007, 2048, 1024, 2048].into_array();
        assert_arrays_eq!(
            array.take(indices.clone())?,
            expected.take(indices)?,
            &mut ctx
        );
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        assert_arrays_eq!(array.cast(dtype.clone())?, expected.cast(dtype)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::before_base([128, 0, 256])]
    #[case::decreasing([0, 256, 128])]
    #[case::misaligned_start([0, 1, 129])]
    #[case::misaligned_end([0, 128, 255])]
    #[case::past_end([0, 128, 4224])]
    #[case::too_wide([0, 0, 1152])]
    fn scalar_rejects_invalid_block(#[case] offsets: [u64; 3]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(ByteBuffer::zeroed(4096)),
            PType::U8,
            Validity::NonNullable,
            None,
            FoR::try_new(PrimitiveArray::from_iter(offsets).into_array(), 0u64.into())?
                .into_array(),
            2048,
            0,
        )?;
        assert!(array.execute_scalar(1024, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn scalar_validates_only_the_selected_block() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let offsets = FoR::try_new(buffer![128u64, 256, 255].into_array(), 0u64.into())?;
        let array = BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(ByteBuffer::zeroed(128)),
            PType::U32,
            Validity::NonNullable,
            None,
            offsets.into_array(),
            2048,
            0,
        )?;
        assert_eq!(array.execute_scalar(0, &mut ctx)?, 0u32.into());
        assert!(array.execute_scalar(1024, &mut ctx).is_err());
        Ok(())
    }

    /// Non-negative values whose 1024-value blocks need different bit widths, with outliers in the
    /// first block.
    fn blocked_values<T: NativePType>(len: usize) -> Vec<T>
    where
        u64: AsPrimitive<T>,
    {
        let max_bits =
            u32::try_from(T::PTYPE.bit_width()).unwrap() - u32::from(T::PTYPE.is_signed_int());
        let widths = [3, 0, max_bits.min(12), 5];
        (0..len)
            .map(|i| {
                let width = if i < FL_CHUNK_SIZE && i % 97 == 0 {
                    max_bits
                } else {
                    widths[(i / FL_CHUNK_SIZE) % widths.len()]
                };
                let mask = 1u64.checked_shl(width).map_or(u64::MAX, |bit| bit - 1);
                (u64::MAX.wrapping_sub(i as u64) & mask).as_()
            })
            .collect()
    }

    #[rstest]
    fn round_trip_best_bit_widths(
        #[values(
            PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
            PType::I64
        )]
        ptype: PType,
        #[values(1, 1024, 4000)] len: usize,
        #[values(false, true)] nullable: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = match_each_integer_ptype!(ptype, |T| {
            let values = blocked_values::<T>(len);
            if nullable {
                PrimitiveArray::from_option_iter(
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(i, value)| (i % 7 != 3).then_some(value)),
                )
            } else {
                PrimitiveArray::from_iter(values)
            }
        });
        let encoded = bitpack_blocked_to_best_bit_widths(&array, &mut ctx)?;
        assert!(matches!(encoded.bit_widths(), BitWidthsView::Blocked(_)));
        assert_arrays_eq!(encoded, array, &mut ctx);
        for index in [0, len / 2, len - 1] {
            assert_eq!(
                encoded.execute_scalar(index, &mut ctx)?,
                array.execute_scalar(index, &mut ctx)?
            );
        }
        Ok(())
    }

    #[rstest]
    fn round_trip_explicit_bit_widths(
        #[values(PType::U8, PType::I16, PType::U32, PType::I64)] ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = match_each_integer_ptype!(ptype, |T| {
            PrimitiveArray::from_iter(blocked_values::<T>(4000))
        });
        // Zero and native widths, plus narrow widths that turn most values into patches.
        let bit_widths = [0, u8::try_from(ptype.bit_width())?, 1, 3];
        let encoded = bitpack_encode_blocked(&array, &bit_widths, None, &mut ctx)?;
        assert_arrays_eq!(encoded, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn for_decodes_blocked_child() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = blocked_values::<u32>(4000);
        let encoded = bitpack_blocked_to_best_bit_widths(
            &PrimitiveArray::from_iter(values.clone()),
            &mut ctx,
        )?;
        let array = FoR::try_new(encoded.into_array(), 1000u32.into())?;
        let expected =
            PrimitiveArray::from_iter(values.into_iter().map(|value| value.wrapping_add(1000)));
        assert_arrays_eq!(array, expected, &mut ctx);
        Ok(())
    }
}
