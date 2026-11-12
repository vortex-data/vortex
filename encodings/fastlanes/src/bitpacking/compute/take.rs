// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PiecewiseSequence;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::chunked_indices;
use super::unpack_chunk_threshold;
use super::unpack_indices_into;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitPackingStrategy;
use crate::unpack_iter::BitUnpackedChunks;

const FULL_ARRAY_DECODE_RATIO: usize = 8;

impl TakeExecute for BitPacked {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * FULL_ARRAY_DECODE_RATIO > array.len() {
            let prim = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return prim.into_array().take(indices.clone()).map(Some);
        }

        if let Some(ranges) =
            contiguous_ranges(indices, array.len(), array.dtype().as_ptype(), ctx)?
        {
            let taken_validity = array.validity()?.take(indices)?;
            let ptype = array.dtype().as_ptype();
            let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
                take_ranges::<T>(array, bit_width, &ranges, indices.len(), taken_validity)
            });
            let taken = taken.reinterpret_cast(ptype);
            let taken = if let Some(patches) = array.patches()
                && let Some(patches) = patches.take(indices, ctx)?
            {
                taken.patch(&patches, ctx)?
            } else {
                taken
            };
            return Ok(Some(taken.into_array()));
        }

        // NOTE: we use the unsigned PType because all values in the BitPackedArray must
        //  be non-negative (pre-condition of creating the BitPackedArray).
        let ptype: PType = PType::try_from(array.dtype())?;
        let validity = array.validity()?;
        let taken_validity = validity.take(indices)?;

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                take_primitive::<T, I>(array, bit_width, &indices, taken_validity, ctx)?
            })
        });
        let taken = if ptype.is_signed_int() {
            PrimitiveArray::from_buffer_handle(
                taken.buffer_handle().clone(),
                ptype,
                taken.validity()?,
            )
        } else {
            taken
        };
        Ok(Some(taken.into_array()))
    }
}

fn contiguous_ranges(
    indices: &ArrayRef,
    array_len: usize,
    ptype: PType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Vec<Range<usize>>>> {
    let Some(indices) = indices.as_opt::<PiecewiseSequence>() else {
        return Ok(None);
    };
    if indices
        .multipliers()
        .as_opt::<Constant>()
        .and_then(|multiplier| multiplier.scalar().as_primitive().as_::<u64>())
        != Some(1)
    {
        return Ok(None);
    }
    let sparse_threshold = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
        unpack_chunk_threshold::<T>() as u64
    });

    let starts = indices
        .starts()
        .cast(PType::U64.into())?
        .execute::<PrimitiveArray>(ctx)?;
    let starts = starts.as_slice::<u64>();
    // The primitive path batches short runs that share a packed chunk, so only take the range
    // path when there is at least one run long enough to unpack a whole chunk.
    match indices.lengths().clone().execute::<Columnar>(ctx)? {
        Columnar::Constant(lengths) => {
            let length = lengths
                .scalar()
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("PiecewiseSequence lengths are non-nullable unsigned integers");
            if length <= sparse_threshold {
                return Ok(None);
            }
            collect_ranges(
                starts.iter().map(|&start| (start, length)),
                array_len,
                indices.len(),
            )
            .map(Some)
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths
                .into_array()
                .cast(PType::U64.into())?
                .execute::<PrimitiveArray>(ctx)?;
            let lengths = lengths.as_slice::<u64>();
            if lengths.iter().all(|&length| length <= sparse_threshold) {
                return Ok(None);
            }
            collect_ranges(
                starts.iter().copied().zip_eq(lengths.iter().copied()),
                array_len,
                indices.len(),
            )
            .map(Some)
        }
    }
}

/// Converts `(start, length)` pieces into validated, non-empty ranges.
fn collect_ranges(
    pieces: impl ExactSizeIterator<Item = (u64, u64)>,
    array_len: usize,
    output_len: usize,
) -> VortexResult<Vec<Range<usize>>> {
    let mut ranges = Vec::with_capacity(pieces.len());
    let mut total_len = 0usize;
    for (start, length) in pieces {
        if length == 0 {
            continue;
        }
        let start = usize::try_from(start)?;
        let length = usize::try_from(length)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        vortex_ensure!(
            end <= array_len,
            "PiecewiseSequenceArray range exceeds array length"
        );
        total_len = total_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
        ranges.push(start..end);
    }
    vortex_ensure_eq!(total_len, output_len);
    Ok(ranges)
}

fn take_ranges<T: NativePType + BitPacking + BitPackedIter<Physical = T>>(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    ranges: &[Range<usize>],
    output_len: usize,
    validity: Validity,
) -> PrimitiveArray {
    let bit_width = usize::from(bit_width);
    let packed = array.packed_slice::<T>();
    let chunk_len = 128 * bit_width / size_of::<T>();
    let offset = usize::from(array.offset());
    let mut output = BufferMut::<T>::with_capacity(output_len);
    let mut scratch = [const { MaybeUninit::uninit() }; FL_CHUNK_SIZE];
    // u64 has the largest sparse decode threshold.
    let mut fragment_indices = [0usize; unpack_chunk_threshold::<u64>()];

    // Vector gathers select runs of components. Long runs are decoded straight into the output so
    // neither the index buffer nor the chunk-grouping work scales with the number of components.
    // Short runs, and the few values a long run selects from a boundary chunk, are cheaper to
    // extract individually. Consecutive short runs are expanded together so that those sharing a
    // chunk are batched.
    let threshold = unpack_chunk_threshold::<T>();
    let mut short_runs_start = 0;
    for (i, range) in ranges.iter().enumerate() {
        if range.len() <= threshold {
            continue;
        }
        take_indices_into(
            &mut output,
            packed,
            bit_width,
            offset,
            ranges[short_runs_start..i].iter().flat_map(Range::clone),
            &mut scratch,
        );
        short_runs_start = i + 1;

        let head_len = (FL_CHUNK_SIZE - (range.start + offset) % FL_CHUNK_SIZE) % FL_CHUNK_SIZE;
        let tail_len = (range.end + offset) % FL_CHUNK_SIZE;
        let middle_start = range.start + if head_len <= threshold { head_len } else { 0 };
        let middle_end =
            (range.end - if tail_len <= threshold { tail_len } else { 0 }).max(middle_start);

        unpack_fragment_into(
            &mut output,
            packed,
            bit_width,
            range.start + offset..middle_start + offset,
            &mut fragment_indices,
        );
        let middle_len = middle_end - middle_start;
        if middle_len != 0 {
            let start = middle_start + offset;
            let first_chunk = start / FL_CHUNK_SIZE;
            let last_chunk = (middle_end + offset).div_ceil(FL_CHUNK_SIZE);
            let mut chunks = BitUnpackedChunks::try_new_with_strategy(
                BitPackingStrategy,
                &packed[first_chunk * chunk_len..last_chunk * chunk_len],
                bit_width,
                start % FL_CHUNK_SIZE,
                middle_len,
                &mut scratch,
            )
            .vortex_expect("validated range lies within the packed buffer");
            let output_start = output.len();
            chunks.decode_into(&mut output.spare_capacity_mut()[..middle_len]);
            // SAFETY: `decode_into` initialized the `middle_len` values following `output_start`.
            unsafe { output.set_len(output_start + middle_len) };
        }
        unpack_fragment_into(
            &mut output,
            packed,
            bit_width,
            middle_end + offset..range.end + offset,
            &mut fragment_indices,
        );
    }
    take_indices_into(
        &mut output,
        packed,
        bit_width,
        offset,
        ranges[short_runs_start..].iter().flat_map(Range::clone),
        &mut scratch,
    );
    PrimitiveArray::new(output, validity)
}

/// Appends the values at `positions`, which lie within one chunk, without decoding the whole chunk.
fn unpack_fragment_into<T: NativePType + BitPacking>(
    output: &mut BufferMut<T>,
    packed: &[T],
    bit_width: usize,
    positions: Range<usize>,
    indices: &mut [usize],
) {
    if positions.is_empty() {
        return;
    }
    let chunk_len = 128 * bit_width / size_of::<T>();
    let packed = &packed[positions.start / FL_CHUNK_SIZE * chunk_len..][..chunk_len];
    let indices = &mut indices[..positions.len()];
    for (index, position) in indices.iter_mut().zip(positions) {
        *index = position % FL_CHUNK_SIZE;
    }
    unpack_indices_into(output, bit_width, packed, indices);
}

/// Appends the values at `indices` to `output`, decoding each visited chunk either fully or only
/// at the selected positions.
fn take_indices_into<T: NativePType + BitPacking>(
    output: &mut BufferMut<T>,
    packed: &[T],
    bit_width: usize,
    offset: usize,
    indices: impl Iterator<Item = usize>,
    unpacked: &mut [MaybeUninit<T>; FL_CHUNK_SIZE],
) {
    let chunk_len = 128 * bit_width / size_of::<T>();
    chunked_indices(indices, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed[chunk_idx * chunk_len..][..chunk_len];

        if indices_within_chunk.len() > unpack_chunk_threshold::<T>() {
            // SAFETY: The validated bit width fits `T`. The source and destination contain one
            // complete FastLanes block. The call initializes every destination value.
            unsafe {
                let dst: &mut [MaybeUninit<T>] = unpacked;
                let dst: &mut [T] = std::mem::transmute(dst);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
            }
            output.extend_trusted(
                indices_within_chunk
                    .iter()
                    // SAFETY: The preceding unpack initialized the complete temporary block.
                    .map(|&index| unsafe { unpacked.get_unchecked(index).assume_init() }),
            );
        } else {
            unpack_indices_into(output, bit_width, packed, indices_within_chunk);
        }
    });
}

fn take_primitive<T: NativePType + BitPacking, I: IntegerPType>(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    indices: &PrimitiveArray,
    taken_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if indices.is_empty() {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), taken_validity));
    }

    let mut output = BufferMut::<T>::with_capacity(indices.len());
    let mut unpacked = [const { MaybeUninit::uninit() }; FL_CHUNK_SIZE];
    take_indices_into(
        &mut output,
        array.packed_slice::<T>(),
        usize::from(bit_width),
        usize::from(array.offset()),
        indices.as_slice::<I>().iter().map(|i| {
            i.to_usize()
                .vortex_expect("index must be expressible as usize")
        }),
        &mut unpacked,
    );

    let unpatched_taken = if array.dtype().as_ptype().is_signed_int() {
        let primitive = PrimitiveArray::new(output, taken_validity);
        PrimitiveArray::from_buffer_handle(
            primitive.buffer_handle().clone(),
            array.dtype().as_ptype(),
            primitive.validity()?,
        )
    } else {
        PrimitiveArray::new(output, taken_validity)
    };
    if let Some(patches) = array.patches()
        && let Some(patches) = patches.take(&indices.clone().into_array(), ctx)?
    {
        return unpatched_taken.patch(&patches, ctx);
    }

    Ok(unpatched_taken)
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::distr::Uniform;
    use rand::rng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PiecewiseSequenceArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::TakeExecute;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;
    use crate::bitpacking::compute::take::take_primitive;
    use crate::bitpacking::compute::take::unpack_chunk_threshold;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn sparse_extraction_covers_batch_and_full_chunk_paths() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        macro_rules! check_type {
            ($T:ty, $bit_width:expr) => {{
                let values = (0..2_048)
                    .map(|index| (index % 127) as $T)
                    .collect::<Vec<_>>();
                let packed = BitPackedData::encode(
                    &PrimitiveArray::from_iter(values.iter().copied()).into_array(),
                    $bit_width,
                    &mut ctx,
                )?;
                let threshold = unpack_chunk_threshold::<$T>();

                for selected in [threshold, threshold + 1] {
                    let indices = (0..selected)
                        .map(|index| (index * 1_024 / selected) as u32)
                        .collect::<Vec<_>>();
                    let actual = take_primitive::<$T, u32>(
                        packed.as_view(),
                        $bit_width,
                        &PrimitiveArray::from_iter(indices.iter().copied()),
                        Validity::NonNullable,
                        &mut ctx,
                    )?;
                    let expected = indices
                        .iter()
                        .map(|&index| values[index as usize])
                        .collect::<Vec<_>>();
                    assert_eq!(actual.as_slice::<$T>(), expected);
                }
            }};
        }

        check_type!(u8, 7);
        check_type!(u16, 15);
        check_type!(u32, 31);
        check_type!(u64, 63);
        Ok(())
    }

    #[test]
    fn sparse_take_preserves_nullable_order_and_duplicates() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..8_192).map(|index| index as u32).collect::<Vec<_>>();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter(values.iter().copied()).into_array(),
            13,
            &mut ctx,
        )?;
        let indices = [
            Some(3_073u32),
            Some(2),
            None,
            Some(2),
            Some(1_025),
            Some(3_073),
            Some(0),
        ];
        let actual = packed
            .take(PrimitiveArray::from_option_iter(indices).into_array())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = PrimitiveArray::from_option_iter(
            indices.map(|index| index.map(|index| values[index as usize])),
        );
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn take_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = buffer![0, 125, 2047, 2049, 2151, 2790].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let primitive_result = bitpacked.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18]),
            &mut ctx
        );
    }

    #[test]
    fn take_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let unpacked = Buffer::from_iter(0u32..1024).into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 2, &mut ctx).unwrap();

        let indices = buffer![0, 2, 4, 6].into_array();

        let primitive_result = bitpacked.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u32, 2, 4, 6]),
            &mut ctx
        );
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = buffer![1919, 1921].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let primitive_result = sliced.take(indices).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([31u8, 33]),
            &mut ctx
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn take_random_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let num_patches: usize = 128;
        let values = (0..u16::MAX as u32 + num_patches as u32).collect::<Buffer<_>>();
        let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let packed = BitPackedData::encode(&uncompressed.into_array(), 16, &mut ctx).unwrap();
        assert!(packed.patches().is_some());

        let rng = rng();
        let range = Uniform::new(0, values.len()).unwrap();
        let random_indices =
            PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));
        let taken = packed.take(random_indices.clone().into_array()).unwrap();

        // sanity check
        random_indices
            .as_slice::<u32>()
            .iter()
            .enumerate()
            .for_each(|(ti, i)| {
                assert_eq!(
                    u32::try_from(&packed.execute_scalar(*i as usize, &mut ctx).unwrap()).unwrap(),
                    values[*i as usize]
                );
                assert_eq!(
                    u32::try_from(&taken.execute_scalar(ti, &mut ctx).unwrap()).unwrap(),
                    values[*i as usize]
                );
            });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn take_signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let start =
            BitPackedData::encode(&buffer![1i32, 2i32, 3i32, 4i32].into_array(), 1, &mut ctx)
                .unwrap();

        let taken_primitive = take_primitive::<u32, u64>(
            start.as_view(),
            1,
            &PrimitiveArray::from_iter([0u64, 1, 2, 3]),
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap();
        assert_arrays_eq!(
            taken_primitive,
            PrimitiveArray::from_iter([1i32, 2, 3, 4]),
            &mut ctx
        );
    }

    #[test]
    fn take_nullable_with_nullables() {
        let mut ctx = SESSION.create_execution_ctx();
        let start =
            BitPackedData::encode(&buffer![1i32, 2i32, 3i32, 4i32].into_array(), 1, &mut ctx)
                .unwrap();

        let taken_primitive = start
            .take(
                PrimitiveArray::from_option_iter([Some(0u64), Some(1), None, Some(3)]).into_array(),
            )
            .unwrap();
        assert_arrays_eq!(
            taken_primitive,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4)]),
            &mut ctx
        );
        let taken_primitive_prim = taken_primitive.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_eq!(taken_primitive_prim.invalid_count(&mut ctx).unwrap(), 1);
    }

    #[rstest]
    #[case(0, 0, false, false, u64::MAX, PType::U64)]
    #[case(7, 0, false, false, u64::MAX, PType::U8)]
    #[case(7, 137, true, true, 0, PType::I32)]
    #[case(15, 1023, false, true, 0, PType::U16)]
    #[case(4, 137, false, true, 0, PType::U8)]
    #[case(4, 1023, true, true, 0, PType::U8)]
    #[case(63, 1023, false, true, 0, PType::U64)]
    #[case(62, 137, true, true, 0, PType::I64)]
    fn test_take_piecewise_ranges(
        #[case] bit_width: u8,
        #[case] slice_offset: usize,
        #[case] patched: bool,
        #[case] nullable: bool,
        #[case] empty_start: u64,
        #[case] ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source_len: usize = 65523;
        let values = PrimitiveArray::from_option_iter((0..source_len).map(|i| {
            if nullable && i.is_multiple_of(29) {
                None
            } else if patched && i.is_multiple_of(31) {
                Some(1u64 << bit_width)
            } else {
                Some((i as u64).wrapping_mul(0x0123_4567_89ab_cdef) & ((1u64 << bit_width) - 1))
            }
        }))
        .into_array()
        .cast(DType::Primitive(ptype, Nullability::Nullable))?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
        let packed = BitPackedData::encode(&values, bit_width, &mut ctx)?;
        let packed =
            BitPacked::slice(packed.as_view(), slice_offset..source_len, &mut ctx)?.unwrap();
        let values = values.slice(slice_offset..source_len)?;
        let indices = PiecewiseSequenceArray::try_new(
            buffer![
                empty_start,
                17,
                991,
                1000,
                3072,
                12000,
                17000,
                17,
                (source_len - slice_offset - 257) as u64
            ]
            .into_array(),
            buffer![0u16, 256, 257, 511, 768, 1024, 1536, 256, 257].into_array(),
            ConstantArray::new(1u8, 9).into_array(),
            4865,
        )?
        .into_array();
        let expanded = indices
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        let actual = BitPacked::take(packed.as_::<BitPacked>(), &indices, &mut ctx)?.unwrap();
        assert_arrays_eq!(actual, values.take(expanded)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_piecewise_fallbacks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..32768).map(|i| i % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        for (length, multiplier, constant) in [
            (7u8, 1u8, true),
            (65, 0, true),
            (65, 2, true),
            (65, 1, false),
        ] {
            let multipliers = if constant {
                ConstantArray::new(multiplier, 2).into_array()
            } else {
                buffer![multiplier, multiplier].into_array()
            };
            for (lengths, len) in [
                (buffer![length, 128].into_array(), usize::from(length) + 128),
                (
                    ConstantArray::new(length, 2).into_array(),
                    usize::from(length) * 2,
                ),
            ] {
                let indices = PiecewiseSequenceArray::try_new(
                    buffer![1017u16, 1023].into_array(),
                    lengths,
                    multipliers.clone(),
                    len,
                )?
                .into_array();
                let expanded = indices
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?
                    .into_array();
                assert_arrays_eq!(packed.take(indices)?, values.take(expanded)?, &mut ctx);
            }
        }
        Ok(())
    }

    #[rstest]
    fn test_take_piecewise_mixed_run_lengths(
        #[values(0, 137)] slice_offset: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0u32..16384).map(|i| i % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        let packed = BitPacked::slice(packed.as_view(), slice_offset..16384, &mut ctx)?.unwrap();
        let values = values.slice(slice_offset..16384)?;
        // Short runs sharing a chunk sit between long runs that cross chunk boundaries. The run at
        // 1900 leaves a short tail in the chunk that the following short run also selects from.
        let indices = PiecewiseSequenceArray::try_new(
            buffer![10u32, 20, 1000, 30, 40, 4000, 50, 4090, 1900, 2070].into_array(),
            buffer![3u32, 5, 300, 2, 0, 2048, 1, 6, 160, 3].into_array(),
            ConstantArray::new(1u8, 10).into_array(),
            2528,
        )?
        .into_array();
        let expanded = indices
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        let actual = BitPacked::take(packed.as_::<BitPacked>(), &indices, &mut ctx)?.unwrap();
        assert_arrays_eq!(actual, values.take(expanded)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(PType::U8, 4, 16)]
    #[case(PType::U16, 9, 32)]
    #[case(PType::U32, 16, 64)]
    #[case(PType::U64, 33, 160)]
    fn test_take_piecewise_sparse_threshold(
        #[case] ptype: PType,
        #[case] bit_width: u8,
        #[case] threshold: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0u64..8192).map(|i| i.wrapping_mul(0x0123_4567_89ab_cdef) & ((1u64 << bit_width) - 1)),
        )
        .into_array()
        .cast(ptype.into())?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
        let packed = BitPackedData::encode(&values, bit_width, &mut ctx)?;
        for length in [threshold, threshold + 1, 2 * threshold + 1] {
            // A run split across two chunks can require sparse extraction on both sides.
            let indices = PiecewiseSequenceArray::try_new(
                buffer![0u64, (1024 - threshold / 2) as u64, 4096].into_array(),
                ConstantArray::new(length as u64, 3).into_array(),
                ConstantArray::new(1u64, 3).into_array(),
                length * 3,
            )?
            .into_array();
            let expanded = indices
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?
                .into_array();
            assert_arrays_eq!(packed.take(indices)?, values.take(expanded)?, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    fn test_take_piecewise_fsl_integration(
        #[values(false, true)] blocked: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (0usize..65536).map(|i| (!i.is_multiple_of(29)).then_some((i % 128) as u8)),
        )
        .into_array();
        let packed = if blocked {
            let widths = (0..64)
                .map(|i| if i % 2 == 0 { 4 } else { 7 })
                .collect::<Vec<_>>();
            BitPackedData::encode_blocked(&values, &widths, &mut ctx)?
        } else {
            BitPackedData::encode(&values, 4, &mut ctx)?
        };
        let validity = Validity::from_iter((0usize..1024).map(|i| !i.is_multiple_of(11)));
        let expected = FixedSizeListArray::new(values, 64, validity.clone(), 1024);
        let array = FixedSizeListArray::new(packed.into_array(), 64, validity, 1024);
        let indices =
            PrimitiveArray::from_option_iter([Some(1000u32), None, Some(0), Some(16), Some(16)])
                .into_array();
        assert_arrays_eq!(
            array.take(indices.clone())?,
            expected.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case(32700, 128, 128)]
    #[case(1, 128, 127)]
    #[case(u64::MAX, 128, 128)]
    fn test_take_piecewise_invalid_range(
        #[case] start: u64,
        #[case] length: u64,
        #[case] declared_len: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..32768).map(|i| i as u32 % 127)).into_array();
        let packed = BitPackedData::encode(&values, 7, &mut ctx)?;
        let indices = PiecewiseSequenceArray::try_new(
            buffer![start].into_array(),
            buffer![length].into_array(),
            ConstantArray::new(1u64, 1).into_array(),
            declared_len,
        )?
        .into_array();
        assert!(
            packed
                .take(indices)?
                .execute::<PrimitiveArray>(&mut ctx)
                .is_err()
        );
        Ok(())
    }

    fn bp(array: vortex_array::ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(&array, bit_width, &mut SESSION.create_execution_ctx()).unwrap()
    }

    #[rstest]
    #[case(bp(PrimitiveArray::from_iter((0..100).map(|i| (i % 63) as u8)).into_array(), 6))]
    #[case(bp(PrimitiveArray::from_iter((0..256).map(|i| i as u32)).into_array(), 8))]
    #[case(bp(buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array(), 3))]
    #[case(bp(
        PrimitiveArray::from_option_iter([Some(10u16), None, Some(20), Some(30), None]).into_array(),
        5
    ))]
    #[case(bp(buffer![42u32].into_array(), 6))]
    #[case(bp(PrimitiveArray::from_iter((0..1024).map(|i| i as u32)).into_array(), 8))]
    fn test_take_bitpacked_conformance(#[case] bitpacked: BitPackedArray) {
        test_take_conformance(&bitpacked.into_array(), &mut SESSION.create_execution_ctx());
    }
}
