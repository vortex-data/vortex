// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::PrimInt;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::bitpack_decompress;

pub fn bitpack_to_best_bit_width(array: &PrimitiveArray) -> VortexResult<BitPackedArray> {
    let bit_width_freq = bit_width_histogram(array.as_view())?;
    let best_bit_width = find_best_bit_width(array.ptype(), &bit_width_freq)?;
    bitpack_encode(array, best_bit_width, Some(&bit_width_freq))
}

#[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
pub fn bitpack_encode(
    array: &PrimitiveArray,
    bit_width: u8,
    bit_width_freq: Option<&[usize]>,
) -> VortexResult<BitPackedArray> {
    let bit_width_freq = match bit_width_freq {
        Some(freq) => freq,
        None => &bit_width_histogram(array.as_view())?,
    };

    // Check array contains no negative values.
    if array.ptype().is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(array.ptype(), |P| {
            array.statistics().compute_min::<P>().unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!(InvalidArgument: "cannot bitpack_encode array containing negative integers")
        }
    }

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
        .then(|| gather_patches(array, bit_width, num_exceptions))
        .transpose()?
        .flatten();

    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        patches,
        bit_width,
        array.len(),
        0,
    )?;
    bitpacked.statistics().inherit_from(array.statistics());
    Ok(bitpacked)
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

    let arr_ref = array.clone().into_array();
    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        None,
        bit_width,
        array.len(),
        0,
    )
    .vortex_expect("bitpacked array construction should succeed");
    bitpacked.statistics().inherit_from(arr_ref.statistics());
    Ok(bitpacked)
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
    let parray = parray.reinterpret_cast(parray.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(parray.ptype(), |P| {
        bitpack_primitive(parray.as_slice::<P>(), bit_width).into_byte_buffer()
    })
}

/// Bitpack a slice of primitives down to the given width.
///
/// See `bitpack` for more caller information.
pub fn bitpack_primitive<T: NativePType + BitPacking>(array: &[T], bit_width: u8) -> Buffer<T> {
    if bit_width == 0 {
        return Buffer::<T>::empty();
    }
    let bit_width = bit_width as usize;

    // How many fastlanes vectors we will process.
    let num_chunks = array.len().div_ceil(1024);
    let num_full_chunks = array.len() / 1024;
    let packed_len = 128 * bit_width / size_of::<T>();
    // packed_len says how many values of size T we're going to include.
    // 1024 * bit_width / 8 == the number of bytes we're going to get.
    // then we divide by the size of T to get the number of elements.

    // Allocate a result byte array.
    let mut output = BufferMut::<T>::with_capacity(num_chunks * packed_len);

    // Loop over all but the last chunk.
    (0..num_full_chunks).for_each(|i| {
        let start_elem = i * 1024;
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &array[start_elem..][..1024],
                &mut output[output_len..][..packed_len],
            );
        };
    });

    // Pad the last chunk with zeros to a full 1024 elements.
    if num_chunks != num_full_chunks {
        let last_chunk_size = array.len() % 1024;
        let mut last_chunk: [T; 1024] = [T::zero(); 1024];
        last_chunk[..last_chunk_size].copy_from_slice(&array[array.len() - last_chunk_size..]);

        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &last_chunk,
                &mut output[output_len..][..packed_len],
            );
        };
    }

    output.freeze()
}

pub fn gather_patches(
    parray: &PrimitiveArray,
    bit_width: u8,
    num_exceptions_hint: usize,
) -> VortexResult<Option<Patches>> {
    let patch_validity = match parray.validity()? {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let array_len = parray.len();
    let validity_mask = parray.validity_mask()?;

    let patches = if array_len < u8::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u8>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u16::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u16>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u32::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u32>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u64>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    };

    Ok(patches)
}

fn gather_patches_impl<T, P>(
    data: &[T],
    bit_width: u8,
    num_exceptions_hint: usize,
    patch_validity: Validity,
    validity_mask: Mask,
) -> VortexResult<Option<Patches>>
where
    T: PrimInt + NativePType,
    P: IntegerPType,
{
    let mut indices: BufferMut<P> = BufferMut::with_capacity(num_exceptions_hint);
    let mut values: BufferMut<T> = BufferMut::with_capacity(num_exceptions_hint);

    let total_chunks = data.len().div_ceil(1024);
    let mut chunk_offsets: BufferMut<u64> = BufferMut::with_capacity(total_chunks);

    for (idx, value) in data.iter().enumerate() {
        if (idx % 1024) == 0 {
            // Record the patch index offset for each chunk.
            chunk_offsets.push(values.len() as u64);
        }

        if (value.leading_zeros() as usize) < T::PTYPE.bit_width() - bit_width as usize
            && validity_mask.value(idx)
        {
            indices.push(P::from(idx).vortex_expect("cast index from usize"));
            values.push(*value);
        }
    }

    if indices.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Patches::new(
            data.len(),
            0,
            indices.into_array(),
            PrimitiveArray::new(values, patch_validity).into_array(),
            Some(chunk_offsets.into_array()),
        )?))
    }
}

pub fn bit_width_histogram(array: ArrayView<'_, Primitive>) -> VortexResult<Vec<usize>> {
    match_each_integer_ptype!(array.ptype(), |P| { bit_width_histogram_typed::<P>(array) })
}

fn bit_width_histogram_typed<T: NativePType + PrimInt>(
    array: ArrayView<'_, Primitive>,
) -> VortexResult<Vec<usize>> {
    let bit_width: fn(T) -> usize =
        |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);

    let mut bit_widths = vec![0usize; size_of::<T>() * 8 + 1];
    match array.validity_mask().bit_buffer() {
        AllOr::All => {
            // All values are valid.
            for v in array.as_slice::<T>() {
                bit_widths[bit_width(*v)] += 1;
            }
        }
        AllOr::None => {
            // All values are invalid
            bit_widths[0] = array.len();
        }
        AllOr::Some(buffer) => {
            // Some values are valid
            for (is_valid, v) in buffer.iter().zip_eq(array.as_slice::<T>()) {
                if is_valid {
                    bit_widths[bit_width(*v)] += 1;
                } else {
                    bit_widths[0] += 1;
                }
            }
        }
    }

    Ok(bit_widths)
}

pub fn find_best_bit_width(ptype: PType, bit_width_freq: &[usize]) -> VortexResult<u8> {
    best_bit_width(bit_width_freq, bytes_per_exception(ptype))
}

/// Assuming exceptions cost 1 value + 1 u32 index, figure out the best bit-width to use.
/// We could try to be clever, but we can never really predict how the exceptions will compress.
#[expect(
    clippy::cast_possible_truncation,
    reason = "bit_width is bounded by check above and result fits in u8"
)]
fn best_bit_width(bit_width_freq: &[usize], bytes_per_exception: usize) -> VortexResult<u8> {
    if bit_width_freq.len() > u8::MAX as usize {
        vortex_bail!("Too many bit widths");
    }

    let len: usize = bit_width_freq.iter().sum();
    let mut num_packed = 0;
    let mut best_cost = len * bytes_per_exception;
    let mut best_width = 0;
    for (bit_width, freq) in bit_width_freq.iter().enumerate() {
        let packed_cost = (bit_width * len).div_ceil(8); // round up to bytes

        num_packed += *freq;
        let exceptions_cost = (len - num_packed) * bytes_per_exception;

        let cost = exceptions_cost + packed_cost;
        if cost < best_cost {
            best_cost = cost;
            best_width = bit_width;
        }
    }

    Ok(best_width as u8)
}

fn bytes_per_exception(ptype: PType) -> usize {
    ptype.byte_width() + 4
}

#[cfg(feature = "_test-harness")]
pub mod test_harness {
    use rand::RngExt;
    use rand::rngs::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;

    use super::bitpack_encode;

    pub fn make_array(
        rng: &mut StdRng,
        len: usize,
        fraction_patches: f64,
        fraction_null: f64,
    ) -> VortexResult<ArrayRef> {
        let values = (0..len)
            .map(|_| {
                let mut v = rng.random_range(0..100i32);
                if rng.random_bool(fraction_patches) {
                    v += 1 << 13
                };
                v
            })
            .collect::<BufferMut<i32>>();

        let values = if fraction_null == 0.0 {
            values.into_array().to_primitive()
        } else {
            let validity = Validity::from_iter((0..len).map(|_| !rng.random_bool(fraction_null)));
            PrimitiveArray::new(values, validity)
        };

        bitpack_encode(&values, 12, None).map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_array::session::ArraySession;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedData;
    use crate::bitpack_compress::test_harness::make_array;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_best_bit_width() {
        // 10 1-bit values, 20 2-bit, etc.
        let freq = vec![0, 10, 20, 15, 1, 0, 0, 0];
        // 3-bits => (46 * 3) + (8 * 1 * 5) => 178 bits => 23 bytes and zero exceptions
        assert_eq!(
            best_bit_width(&freq, bytes_per_exception(PType::U8)).unwrap(),
            3
        );
    }

    #[test]
    fn null_patches() {
        let valid_values = (0..24).map(|v| v < 1 << 4).collect::<Vec<_>>();
        let values = PrimitiveArray::new(
            (0u32..24).collect::<Buffer<_>>(),
            Validity::from_iter(valid_values),
        );
        assert!(values.ptype().is_unsigned_int());
        let compressed = BitPackedData::encode(&values.into_array(), 4).unwrap();
        assert!(compressed.patches().is_none());
        assert_eq!(
            (0..(1 << 4)).collect::<Vec<_>>(),
            compressed
                .validity_mask()
                .unwrap()
                .to_bit_buffer()
                .set_indices()
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn compress_signed_fails() {
        let values: Buffer<i64> = (-500..500).collect();
        let array = PrimitiveArray::new(values, Validity::AllValid);
        assert!(array.ptype().is_signed_int());

        let err = BitPackedData::encode(&array.into_array(), 1024u32.ilog2() as u8).unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }

    #[test]
    fn canonicalize_chunked_of_bitpacked() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(0);

        let chunks = (0..10)
            .map(|_| make_array(&mut rng, 100, 0.25, 0.25).unwrap())
            .collect::<Vec<_>>();
        let chunked = ChunkedArray::from_iter(chunks).into_array();

        let into_ca = chunked.to_primitive();
        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        chunked.append_to_builder(&mut primitive_builder, &mut SESSION.create_execution_ctx())?;
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into);

        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        primitive_builder.extend_from_array(&chunked);
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into);

        Ok(())
    }

    #[test]
    fn test_chunk_offsets() {
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 3000, 3100];
        let mut values = vec![0u32; 4096usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();

        let patches = bitpacked.patches().unwrap();
        let chunk_offsets = patches.chunk_offsets().as_ref().unwrap().to_primitive();

        // chunk 0 (0-1023): patches at 100, 200 -> starts at patch index 0
        // chunk 1 (1024-2047): no patches -> points to patch index 2
        // chunk 2 (2048-3071): patch at 3000 -> starts at patch index 2
        // chunk 3 (3072-4095): patch at 3100 -> starts at patch index 3
        assert_arrays_eq!(chunk_offsets, PrimitiveArray::from_iter([0u64, 2, 2, 3]));
    }

    #[test]
    fn test_chunk_offsets_no_patches_in_middle() {
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 2500];
        let mut values = vec![0u32; 3072usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();

        let patches = bitpacked.patches().unwrap();
        let chunk_offsets = patches.chunk_offsets().as_ref().unwrap().to_primitive();

        assert_arrays_eq!(chunk_offsets, PrimitiveArray::from_iter([0u64, 2, 2]));
    }

    #[test]
    fn test_chunk_offsets_trailing_empty_chunks() {
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200, 1500];
        let mut values = vec![0u32; 5120usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();

        let patches = bitpacked.patches().unwrap();
        let chunk_offsets = patches.chunk_offsets().as_ref().unwrap().to_primitive();

        // chunk 0 (0-1023): patches at 100, 200 -> starts at patch index 0
        // chunk 1 (1024-2047): patch at 1500 -> starts at patch index 2
        // chunk 2 (2048-3071): no patches -> points to patch index 3
        // chunk 3 (3072-4095): no patches -> points to patch index 3 (remaining chunks filled)
        // chunk 4 (4096-5119): no patches -> points to patch index 3 (remaining chunks filled)
        assert_arrays_eq!(chunk_offsets, PrimitiveArray::from_iter([0u64, 2, 3, 3, 3]));
    }

    #[test]
    fn test_chunk_offsets_single_chunk() {
        let patch_value = 1u32 << 20;
        let patch_indices = [100usize, 200];
        let mut values = vec![0u32; 500usize];

        patch_indices
            .iter()
            .for_each(|&idx| values[idx] = patch_value);

        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();

        let patches = bitpacked.patches().unwrap();
        let chunk_offsets = patches.chunk_offsets().as_ref().unwrap().to_primitive();

        // Single chunk starting at patch index 0.
        assert_arrays_eq!(chunk_offsets, PrimitiveArray::from_iter([0u64]));
    }
}
