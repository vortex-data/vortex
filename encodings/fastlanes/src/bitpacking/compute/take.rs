// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
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

use super::chunked_indices;
use super::unpack_chunk_threshold;
use super::unpack_indices_into;
use crate::BitPacked;
use crate::BitPackedArrayExt;
const FULL_ARRAY_DECODE_RATIO: usize = 8;

impl TakeExecute for BitPacked {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * FULL_ARRAY_DECODE_RATIO > array.len() {
            let prim = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return prim.into_array().take(indices.clone()).map(Some);
        }

        // NOTE: we use the unsigned PType because all values in the BitPackedArray must
        //  be non-negative (pre-condition of creating the BitPackedArray).
        let ptype: PType = PType::try_from(array.dtype())?;
        let validity = array.validity()?;
        let taken_validity = validity.take(indices)?;

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                take_primitive::<T, I>(array, &indices, taken_validity, ctx)?
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

fn take_primitive<T: NativePType + BitPacking, I: IntegerPType>(
    array: ArrayView<'_, BitPacked>,
    indices: &PrimitiveArray,
    taken_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if indices.is_empty() {
        return Ok(PrimitiveArray::new(Buffer::<T>::empty(), taken_validity));
    }

    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;

    let packed = array.packed_slice::<T>();

    // Group indices by 1024-element chunk, *without* allocating on the heap
    let indices_iter = indices.as_slice::<I>().iter().map(|i| {
        i.to_usize()
            .vortex_expect("index must be expressible as usize")
    });

    let mut output = BufferMut::<T>::with_capacity(indices.len());
    let mut unpacked = [const { MaybeUninit::uninit() }; 1024];
    let chunk_len = 128 * bit_width / size_of::<T>();

    chunked_indices(indices_iter, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed[chunk_idx * chunk_len..][..chunk_len];

        if indices_within_chunk.len() > unpack_chunk_threshold::<T>() {
            // SAFETY: The validated bit width fits `T`. The source and destination contain one
            // complete FastLanes block. The call initializes every destination value.
            unsafe {
                let dst: &mut [MaybeUninit<T>] = &mut unpacked;
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
            unpack_indices_into(&mut output, bit_width, packed, indices_within_chunk);
        }
    });

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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedArray;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;
    use crate::bitpacking::compute::take::take_primitive;
    use crate::bitpacking::compute::take::unpack_chunk_threshold;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
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
