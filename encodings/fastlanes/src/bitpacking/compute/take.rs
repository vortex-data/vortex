// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
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
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;

use super::chunked_indices;
use crate::BitPacked;
use crate::BitPackedArray;
use crate::bitpack_decompress::unpack_single;
use crate::bitpack_decompress::unpack_single_primitive;

// TODO(connor): This is duplicated in `encodings/fastlanes/src/bitpacking/kernels/mod.rs`.
/// assuming the buffer is already allocated (which will happen at most once) then unpacking
/// all 1024 elements takes ~8.8x as long as unpacking a single element on an M2 Macbook Air.
/// see https://github.com/vortex-data/vortex/pull/190#issue-2223752833
pub(super) const UNPACK_CHUNK_THRESHOLD: usize = 8;

impl TakeExecute for BitPacked {
    fn take(
        array: &BitPackedArray,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the indices are large enough, it's faster to flatten and take the primitive array.
        if indices.len() * UNPACK_CHUNK_THRESHOLD > array.len() {
            let prim = array.clone().into_array().execute::<PrimitiveArray>(ctx)?;
            return prim.take(indices.to_array()).map(Some);
        }

        // NOTE: we use the unsigned PType because all values in the BitPackedArray must
        //  be non-negative (pre-condition of creating the BitPackedArray).
        let ptype: PType = PType::try_from(array.dtype())?;
        let validity = array.validity();
        let taken_validity = validity.take(indices)?;

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let taken = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                take_primitive::<T, I>(array, &indices, taken_validity, ctx)?
            })
        });
        Ok(Some(taken.reinterpret_cast(ptype).into_array()))
    }
}

fn take_primitive<T: NativePType + BitPacking, I: IntegerPType>(
    array: &BitPackedArray,
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

        let mut have_unpacked = false;
        let mut offset_chunk_iter = indices_within_chunk.chunks_exact(UNPACK_CHUNK_THRESHOLD);

        // this loop only runs if we have at least UNPACK_CHUNK_THRESHOLD offsets
        for offset_chunk in &mut offset_chunk_iter {
            assert_eq!(offset_chunk.len(), UNPACK_CHUNK_THRESHOLD); // let compiler know slice length
            if !have_unpacked {
                unsafe {
                    let dst: &mut [MaybeUninit<T>] = &mut unpacked;
                    let dst: &mut [T] = mem::transmute(dst);
                    BitPacking::unchecked_unpack(bit_width, packed, dst);
                }
                have_unpacked = true;
            }

            for &index in offset_chunk {
                output.push(unsafe { unpacked[index].assume_init() });
            }
        }

        // if we have a remainder (i.e., < UNPACK_CHUNK_THRESHOLD leftover offsets), we need to handle it
        if !offset_chunk_iter.remainder().is_empty() {
            if have_unpacked {
                // we already bulk unpacked this chunk, so we can just push the remaining elements
                for &index in offset_chunk_iter.remainder() {
                    output.push(unsafe { unpacked[index].assume_init() });
                }
            } else {
                // we had fewer than UNPACK_CHUNK_THRESHOLD offsets in the first place,
                // so we need to unpack each one individually
                for &index in offset_chunk_iter.remainder() {
                    output.push(unsafe { unpack_single_primitive::<T>(packed, bit_width, index) });
                }
            }
        }
    });

    let mut unpatched_taken = PrimitiveArray::new(output, taken_validity);
    // Flip back to signed type before patching.
    if array.ptype().is_signed_int() {
        unpatched_taken = unpatched_taken.reinterpret_cast(array.ptype());
    }

    Ok(unpatched_taken)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use rand::RngExt;
    use rand::distr::Uniform;
    use rand::rng;
    use rstest::rstest;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::BitPackedArray;
    use crate::bitpack_compress::BitPackEncoder;
    use crate::bitpacking::compute::take::take_primitive;

    #[test]
    fn take_indices() {
        let indices = buffer![0, 125, 2047, 2049, 2151, 2790].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackEncoder::new(&unpacked)
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();

        let primitive_result = bitpacked.take(indices.to_array()).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18])
        );
    }

    #[test]
    fn take_with_patches() {
        let unpacked = PrimitiveArray::from_iter(0u32..1024);
        let bitpacked = BitPackEncoder::new(&unpacked)
            .with_bit_width(2)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();

        let indices = buffer![0, 2, 4, 6].into_array();

        let primitive_result = bitpacked.take(indices.to_array()).unwrap();
        assert_arrays_eq!(primitive_result, PrimitiveArray::from_iter([0u32, 2, 4, 6]));
    }

    #[test]
    fn take_sliced_indices() {
        let indices = buffer![1919, 1921].into_array();

        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackEncoder::new(&unpacked)
            .with_bit_width(6)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let primitive_result = sliced.take(indices.to_array()).unwrap();
        assert_arrays_eq!(primitive_result, PrimitiveArray::from_iter([31u8, 33]));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn take_random_indices() {
        let num_patches: usize = 128;
        let values = (0..u16::MAX as u32 + num_patches as u32).collect::<Buffer<_>>();
        let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let packed = BitPackEncoder::new(&uncompressed)
            .with_bit_width(16)
            .pack()
            .unwrap();
        assert!(packed.has_patches());

        let packed = packed.into_array().unwrap();

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
                    u32::try_from(&packed.scalar_at(*i as usize).unwrap()).unwrap(),
                    values[*i as usize]
                );
                assert_eq!(
                    u32::try_from(&taken.scalar_at(ti).unwrap()).unwrap(),
                    values[*i as usize]
                );
            });
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn take_signed_with_patches() {
        let primitive = PrimitiveArray::from_iter([1i32, 2i32, 3i32, 4i32]);
        let start = BitPackEncoder::new(&primitive)
            .with_bit_width(1)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();

        let taken_primitive = start
            .take(PrimitiveArray::from_iter([0u64, 1, 2, 3]).into_array())
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(taken_primitive, PrimitiveArray::from_iter([1i32, 2, 3, 4]));
    }

    #[test]
    fn take_nullable_with_nullables() {
        let primitive = PrimitiveArray::from_iter([1i32, 2i32, 3i32, 4i32]);
        let start = BitPackEncoder::new(&primitive)
            .with_bit_width(1)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();

        let taken_primitive = start
            .take(
                PrimitiveArray::from_option_iter([Some(0u64), Some(1), None, Some(3)]).into_array(),
            )
            .unwrap();
        assert_arrays_eq!(
            taken_primitive,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4)])
        );
        assert_eq!(taken_primitive.to_primitive().invalid_count().unwrap(), 1);
    }

    #[rstest]
    #[case((0..100).map(|i| Some((i % 63) as u8)), 6)]
    #[case((0..256).map(|i| Some(i as u32)), 8)]
    #[case((1i32..=8).map(|i| Some(i)), 3)]
    #[case([Some(10u16), None, Some(20), Some(30), None], 5)]
    #[case([Some(42u32)], 6)]
    #[case((0..1024).map(|i| Some(i as u32)), 8)]
    fn test_take_bitpacked_conformance<T: NativePType>(
        #[case] values: impl IntoIterator<Item = Option<T>>,
        #[case] bit_width: u8,
    ) {
        use vortex_array::compute::conformance::take::test_take_conformance;
        let parray = PrimitiveArray::from_option_iter(values);
        let bitpacked = BitPackEncoder::new(&parray)
            .with_bit_width(bit_width)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        println!("BITPACKED: {:?}", bitpacked.encoding_id());
        test_take_conformance(&bitpacked);
    }
}
