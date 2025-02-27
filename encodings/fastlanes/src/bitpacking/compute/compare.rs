use arrow_buffer::{BooleanBuffer, bit_util};
use fastlanes::BitPackingCompare;
use vortex_array::arrays::BoolArray;
use vortex_array::compute::{CompareFn, Operator};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef};
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, PType};
use vortex_error::{VortexError, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{BitPackedArray, BitPackedEncoding};

impl CompareFn<&BitPackedArray> for BitPackedEncoding {
    fn compare(
        &self,
        lhs: &BitPackedArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if lhs.patches().is_some() {
            return Ok(None);
        }

        if lhs.dtype.is_nullable() || rhs.dtype().is_nullable() {
            return Ok(None);
        };

        // TODO(joe): support sliced files
        if lhs.offset != 0 {
            return Ok(None);
        }

        let Some(rhs) = rhs.as_constant() else {
            return Ok(None);
        };

        let validity = lhs.validity.mask(&if rhs.is_null() {
            Mask::AllTrue(lhs.len())
        } else {
            Mask::AllFalse(lhs.len())
        })?;

        // println!("a {}", lhs.to_array().tree_display());

        let res = match lhs.ptype() {
            PType::U32 => compare_impl::<u32>(lhs, rhs, operator),
            PType::U64 => compare_impl::<u64>(lhs, rhs, operator),
            PType::I32 => compare_impl::<i32>(lhs, rhs, operator),
            PType::I64 => compare_impl::<i64>(lhs, rhs, operator),
            _ => return Ok(None),
        };

        Ok(res?.map(|buffer| BoolArray::new(buffer, validity).into_array()))
        // match_each_integer_ptype!(lhs.ptype(), |$P| {
        //     Ok(compare_impl::<$P>(lhs, rhs, operator)?
        //         .map(|buffer| BoolArray::new(buffer, validity).into_array()))
        // })
    }
}

fn compare_impl<T>(
    array: &BitPackedArray,
    scalar: Scalar,
    operator: Operator,
) -> VortexResult<Option<BooleanBuffer>>
where
    T: NativePType + fastlanes::FastLanesComparable + TryFrom<Scalar, Error = VortexError>,
    T::Bitpacked: NativePType + BitPackingCompare,
{
    const CHUNK_SIZE: usize = 1024;
    if array.bit_width() == 0 {
        return Ok(Some(BooleanBuffer::new_unset(array.len())));
    }

    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;

    let last_chunk_length = if (offset + array.len()) % CHUNK_SIZE == 0 {
        CHUNK_SIZE
    } else {
        (offset + array.len()) % CHUNK_SIZE
    };

    let output_len = bit_util::round_upto_multiple_of_64(array.len()) / 64;
    let mut output = BufferMut::<u64>::with_capacity(output_len);
    let packed = array.packed_slice::<T::Bitpacked>();

    // How many fastlanes vectors we will process.
    // Packed array might not start at 0 when the array is sliced. Offset is guaranteed to be < 1024.
    let num_chunks = (offset + array.len()).div_ceil(CHUNK_SIZE);
    let elems_per_chunk = 128 * bit_width / size_of::<T>();

    let first_chunk_is_sliced = offset != 0;
    let last_chunk_is_sliced = last_chunk_length != CHUNK_SIZE;
    let full_chunks_range =
        (first_chunk_is_sliced as usize)..(num_chunks - last_chunk_is_sliced as usize);

    let value = T::try_from(scalar)?;

    for i in full_chunks_range.clone() {
        let chunk = &packed[i * elems_per_chunk..][..elems_per_chunk];

        unsafe {
            // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
            unchecked_unpack_cmp_impl::<T>(
                bit_width,
                chunk,
                // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
                &mut *(output.spare_capacity_mut().assume_init_mut()[i * 16..(i + 1) * 16]
                    .as_mut_ptr() as *mut [u64; 16]),
                operator,
                value,
            )
        }
    }
    unsafe { output.set_len(full_chunks_range.len() * 16) }

    if last_chunk_is_sliced {
        let chunk = &packed[(num_chunks - 1) * elems_per_chunk..][..elems_per_chunk];
        let mut decoded = [0u64; CHUNK_SIZE / u64::BITS as usize];
        // SAFETY:
        // 1. chunk is elems_per_chunk.
        // 2. decoded is exactly 1024.
        unsafe { unchecked_unpack_cmp_impl::<T>(bit_width, chunk, &mut decoded, operator, value) };
        output.spare_capacity_mut()[0..last_chunk_length.div_ceil(16)]
            .write_copy_of_slice(&decoded[..last_chunk_length.div_ceil(16)]);
        unsafe {
            output.set_len(full_chunks_range.len() * 16 + last_chunk_length.div_ceil(16));
        }
    }

    // TODO(joe): fix conversion
    Ok(Some(BooleanBuffer::new(
        arrow_buffer::Buffer::from(output.to_vec()),
        0,
        array.len(),
    )))
}

// unsafe fn unchecked_unpack_cmp_impl<T: NativePType + fastlanes::FastLanesComparable>(
//     _width: usize,
//     _input: &[T::Bitpacked],
//     _output: &mut [u64; 16],
//     _comparison: Operator,
//     _value: T,
// ) where
//     T::Bitpacked: NativePType + BitPackingCompare,
// {
//     todo!()
// }

unsafe fn unchecked_unpack_cmp_impl<T: NativePType + fastlanes::FastLanesComparable>(
    width: usize,
    input: &[T::Bitpacked],
    output: &mut [u64; 16],
    comparison: Operator,
    value: T,
) where
    T::Bitpacked: NativePType + BitPackingCompare,
{
    match comparison {
        Operator::Eq => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a == b, value)
        },
        Operator::NotEq => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a != b, value)
        },
        Operator::Gte => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a >= b, value)
        },
        Operator::Gt => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a > b, value)
        },
        Operator::Lte => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a <= b, value)
        },
        Operator::Lt => unsafe {
            T::Bitpacked::unchecked_unpack_cmp(width, input, output, |a, b| a < b, value)
        },
    }
}

// impl CompareFn<&ConstantArray> for ConstantEncoding {

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::{CompareFn, Operator};
    use vortex_array::{Array, ToCanonical};

    use crate::{BitPackedEncoding, bitpack_encode};

    #[test]
    fn test_compare() {
        let arr = PrimitiveArray::from_iter(0u32..1026);
        let arr = bitpack_encode(&arr, 11).unwrap();

        println!("arr {}", arr.clone().to_array().tree_display());

        let const_ = ConstantArray::new(1025u32, arr.len()).into_array();

        let res = BitPackedEncoding::compare(&BitPackedEncoding, &arr, &const_, Operator::Lt)
            .unwrap()
            .unwrap();
        let vec = res.to_bool().unwrap().boolean_buffer().iter().collect_vec();
        println!(
            "res {:?}, {:?}, {:?}",
            vec.len(),
            vec.iter().map(|x| *x as usize).sum::<usize>(),
            vec.iter().map(|x| 1 - *x as usize).sum::<usize>()
        );
    }
}
