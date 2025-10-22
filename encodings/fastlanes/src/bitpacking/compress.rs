// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::PrimInt;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::{ArrayBuilder as _, PrimitiveBuilder, UninitRange};
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{
    IntegerPType, NativePType, PType, PhysicalPType, match_each_integer_ptype,
    match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::BitPackedArray;
use crate::unpack_iter::{BitPacked, UnpackStrategy};

pub fn bitpack_to_best_bit_width(array: &PrimitiveArray) -> VortexResult<BitPackedArray> {
    let bit_width_freq = bit_width_histogram(array)?;
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
        None => &bit_width_histogram(array)?,
    };

    // Check array contains no negative values.
    if array.ptype().is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(array.ptype(), |P| {
            array.statistics().compute_min::<P>().unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!("cannot bitpack_encode array containing negative integers")
        }
    }

    let num_exceptions = count_exceptions(bit_width, bit_width_freq);

    if bit_width >= array.ptype().bit_width() as u8 {
        // Nothing we can do
        vortex_bail!(
            "Cannot pack - specified bit width {bit_width} >= {}",
            array.ptype().bit_width()
        )
    }

    // SAFETY: we check that array only contains non-negative values.
    let packed = unsafe { bitpack_unchecked(array, bit_width)? };
    let patches = (num_exceptions > 0)
        .then(|| gather_patches(array, bit_width, num_exceptions))
        .transpose()?
        .flatten();

    // SAFETY: all components validated above
    unsafe {
        Ok(BitPackedArray::new_unchecked(
            packed,
            array.dtype().clone(),
            array.validity().clone(),
            patches,
            bit_width,
            array.len(),
            0,
        ))
    }
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
    let packed = unsafe { bitpack_unchecked(&array, bit_width)? };

    // SAFETY: checked by bitpack_unchecked
    unsafe {
        Ok(BitPackedArray::new_unchecked(
            packed,
            array.dtype().clone(),
            array.validity().clone(),
            None,
            bit_width,
            array.len(),
            0,
        ))
    }
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
pub unsafe fn bitpack_unchecked(
    parray: &PrimitiveArray,
    bit_width: u8,
) -> VortexResult<ByteBuffer> {
    let parray = parray.reinterpret_cast(parray.ptype().to_unsigned());
    let packed = match_each_unsigned_integer_ptype!(parray.ptype(), |P| {
        bitpack_primitive(parray.as_slice::<P>(), bit_width).into_byte_buffer()
    });
    Ok(packed)
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
    let patch_validity = match parray.validity() {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let array_len = parray.len();
    let validity_mask = parray.validity_mask();

    let patches = if array_len < u8::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u8>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )
        })
    } else if array_len < u16::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u16>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )
        })
    } else if array_len < u32::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u32>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )
        })
    } else {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u64>(
                parray.as_slice::<T>(),
                bit_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )
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
) -> Option<Patches>
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

    (!indices.is_empty()).then(|| {
        Patches::new(
            data.len(),
            0,
            indices.into_array(),
            PrimitiveArray::new(values, patch_validity).into_array(),
            Some(chunk_offsets.into_array()),
        )
    })
}

/// BitPacking strategy - uses plain bitpacking without reference value
pub struct BitPackingStrategy;

impl<T: PhysicalPType<Physical: BitPacking>> UnpackStrategy<T> for BitPackingStrategy {
    #[inline(always)]
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: Caller must ensure [`BitPacking::unchecked_unpack`] safety requirements hold.
        unsafe {
            BitPacking::unchecked_unpack(bit_width, chunk, dst);
        }
    }
}

pub fn unpack(array: &BitPackedArray) -> PrimitiveArray {
    match_each_integer_ptype!(array.ptype(), |P| { unpack_primitive::<P>(array) })
}

pub fn unpack_primitive<T: BitPacked>(array: &BitPackedArray) -> PrimitiveArray {
    let mut builder = PrimitiveBuilder::with_capacity(array.dtype().nullability(), array.len());
    unpack_into::<T>(array, &mut builder);
    assert_eq!(builder.len(), array.len());
    builder.finish_into_primitive()
}

pub(crate) fn unpack_into<T: BitPacked>(
    array: &BitPackedArray,
    // TODO(ngates): do we want to use fastlanes alignment for this buffer?
    builder: &mut PrimitiveBuilder<T>,
) {
    // If the array is empty, then we don't need to add anything to the builder.
    if array.is_empty() {
        return;
    }

    let mut uninit_range = builder.uninit_range(array.len());

    // SAFETY: We later initialize the the uninitialized range of values with `copy_from_slice`.
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(array.validity_mask());
    }

    let mut bit_packed_iter = array.unpacked_chunks();
    bit_packed_iter.decode_into(&mut uninit_range);

    if let Some(patches) = array.patches() {
        apply_patches(&mut uninit_range, patches);
    };

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via calls to `copy_from_slice`.
    unsafe {
        uninit_range.finish();
    }
}

pub fn apply_patches<T: NativePType>(dst: &mut UninitRange<T>, patches: &Patches) {
    apply_patches_fn(dst, patches, |x| x)
}

pub fn apply_patches_fn<T: NativePType, F: Fn(T) -> T>(
    dst: &mut UninitRange<T>,
    patches: &Patches,
    f: F,
) {
    assert_eq!(patches.array_len(), dst.len());

    let indices = patches.indices().to_primitive();
    let values = patches.values().to_primitive();
    let validity = values.validity_mask();
    let values = values.as_slice::<T>();

    match_each_unsigned_integer_ptype!(indices.ptype(), |P| {
        insert_values_and_validity_at_indices(
            dst,
            indices.as_slice::<P>(),
            values,
            validity,
            patches.offset(),
            f,
        )
    });
}

fn insert_values_and_validity_at_indices<T: NativePType, IndexT: IntegerPType, F: Fn(T) -> T>(
    dst: &mut UninitRange<T>,
    indices: &[IndexT],
    values: &[T],
    values_validity: Mask,
    indices_offset: usize,
    f: F,
) {
    match values_validity {
        Mask::AllTrue(_) => {
            for (index, &value) in indices.iter().zip_eq(values) {
                dst.set_value(index.as_() - indices_offset, f(value));
            }
        }
        Mask::AllFalse(_) => {
            for decompressed_index in indices {
                dst.set_validity_bit(decompressed_index.as_() - indices_offset, false);
            }
        }
        Mask::Values(vb) => {
            for (index, &value) in indices.iter().zip_eq(values) {
                let out_index = index.as_() - indices_offset;
                dst.set_value(out_index, f(value));
                dst.set_validity_bit(out_index, vb.value(out_index));
            }
        }
    }
}

pub fn unpack_single(array: &BitPackedArray, index: usize) -> Scalar {
    let bit_width = array.bit_width() as usize;
    let ptype = array.ptype();
    // let packed = array.packed().into_primitive()?;
    let index_in_encoded = index + array.offset() as usize;
    let scalar: Scalar = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |P| {
        unsafe {
            unpack_single_primitive::<P>(array.packed_slice::<P>(), bit_width, index_in_encoded)
                .into()
        }
    });
    // Cast to fix signedness and nullability
    scalar.cast(array.dtype()).vortex_expect("cast failure")
}

/// # Safety
///
/// The caller must ensure the following invariants hold:
/// * `packed.len() == (length + 1023) / 1024 * 128 * bit_width`
/// * `index_to_decode < length`
///
/// Where `length` is the length of the array/slice backed by `packed`
/// (but is not provided to this function).
pub unsafe fn unpack_single_primitive<T: NativePType + BitPacking>(
    packed: &[T],
    bit_width: usize,
    index_to_decode: usize,
) -> T {
    let chunk_index = index_to_decode / 1024;
    let index_in_chunk = index_to_decode % 1024;
    let elems_per_chunk: usize = 128 * bit_width / size_of::<T>();

    let packed_chunk = &packed[chunk_index * elems_per_chunk..][0..elems_per_chunk];
    unsafe { BitPacking::unchecked_unpack_single(bit_width, packed_chunk, index_in_chunk) }
}

pub fn find_best_bit_width(ptype: PType, bit_width_freq: &[usize]) -> VortexResult<u8> {
    best_bit_width(bit_width_freq, bytes_per_exception(ptype))
}

/// Assuming exceptions cost 1 value + 1 u32 index, figure out the best bit-width to use.
/// We could try to be clever, but we can never really predict how the exceptions will compress.
#[allow(clippy::cast_possible_truncation)]
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

pub fn count_exceptions(bit_width: u8, bit_width_freq: &[usize]) -> usize {
    if bit_width_freq.len() <= bit_width as usize {
        return 0;
    }
    bit_width_freq[bit_width as usize + 1..].iter().sum()
}

pub fn bit_width_histogram(array: &PrimitiveArray) -> VortexResult<Vec<usize>> {
    match_each_integer_ptype!(array.ptype(), |P| { bit_width_histogram_typed::<P>(array) })
}

fn bit_width_histogram_typed<T: NativePType + PrimInt>(
    array: &PrimitiveArray,
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

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use rand::Rng as _;
    use rand::rngs::StdRng;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayRef, IntoArray, ToCanonical};
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
#[allow(clippy::cast_possible_truncation)]
mod test {
    use rand::SeedableRng as _;
    use rand::rngs::StdRng;
    use vortex_array::ToCanonical as _;
    use vortex_array::arrays::ChunkedArray;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::Nullability;
    use vortex_error::VortexError;

    use super::*;
    use crate::BitPackedVTable;
    use crate::bitpacking::compress::test_harness::make_array;

    #[test]
    fn test_all_zeros() {
        let zeros = buffer![0u16, 0, 0, 0].into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 0, None).unwrap();
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &[0u16, 0, 0, 0]);
    }

    #[test]
    fn test_simple_patches() {
        let zeros = buffer![0u16, 1, 0, 1].into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 0, None).unwrap();
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &[0u16, 1, 0, 1]);
    }

    #[test]
    fn test_one_full_chunk() {
        let zeros = BufferMut::from_iter(0u16..1024).into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &(0u16..1024).collect::<Vec<_>>());
    }

    #[test]
    fn test_three_full_chunks_with_patches() {
        let zeros = BufferMut::from_iter((5u16..1029).chain(5u16..1029).chain(5u16..1029))
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert!(bitpacked.patches().is_some());
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(
            actual,
            &(5u16..1029)
                .chain(5u16..1029)
                .chain(5u16..1029)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_no_patch() {
        let zeros = BufferMut::from_iter(0u16..1025).into_array().to_primitive();
        let bitpacked = bitpack_encode(&zeros, 11, None).unwrap();
        assert!(bitpacked.patches().is_none());
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &(0u16..1025).collect::<Vec<_>>());
    }

    #[test]
    fn test_one_full_chunk_and_one_short_chunk_with_patches() {
        let zeros = BufferMut::from_iter(512u16..1537)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 1025);
        assert!(bitpacked.patches().is_some());
        let actual = unpack(&bitpacked);
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &(512u16..1537).collect::<Vec<_>>());
    }

    #[test]
    fn test_offset_and_short_chunk_and_patches() {
        let zeros = BufferMut::from_iter(512u16..1537)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 1025);
        assert!(bitpacked.patches().is_some());
        let bitpacked = bitpacked.slice(1023..1025);
        let actual = unpack(bitpacked.as_::<BitPackedVTable>());
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &[1535, 1536]);
    }

    #[test]
    fn test_offset_and_short_chunk_with_chunks_between_and_patches() {
        let zeros = BufferMut::from_iter(512u16..2741)
            .into_array()
            .to_primitive();
        let bitpacked = bitpack_encode(&zeros, 10, None).unwrap();
        assert_eq!(bitpacked.len(), 2229);
        assert!(bitpacked.patches().is_some());
        let bitpacked = bitpacked.slice(1023..2049);
        let actual = unpack(bitpacked.as_::<BitPackedVTable>());
        let actual = actual.as_slice::<u16>();
        assert_eq!(actual, &(1023..2049).map(|x| x + 512).collect::<Vec<_>>());
    }

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
        let compressed = BitPackedArray::encode(values.as_ref(), 4).unwrap();
        assert!(compressed.patches().is_none());
        assert_eq!(
            (0..(1 << 4)).collect::<Vec<_>>(),
            compressed
                .validity_mask()
                .to_bit_buffer()
                .set_indices()
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn test_compression_roundtrip_fast() {
        compression_roundtrip(125);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn test_compression_roundtrip() {
        compression_roundtrip(1024);
        compression_roundtrip(10_000);
        compression_roundtrip(10_240);
    }

    fn compression_roundtrip(n: usize) {
        let values = PrimitiveArray::from_iter((0..n).map(|i| (i % 2047) as u16));
        let compressed = BitPackedArray::encode(values.as_ref(), 11).unwrap();
        let decompressed = compressed.to_primitive();
        assert_eq!(decompressed.as_slice::<u16>(), values.as_slice::<u16>());

        values
            .as_slice::<u16>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                let scalar: u16 = unpack_single(&compressed, i).try_into().unwrap();
                assert_eq!(scalar, *v);
            });
    }

    #[test]
    fn compress_signed_fails() {
        let values: Buffer<i64> = (-500..500).collect();
        let array = PrimitiveArray::new(values, Validity::AllValid);
        assert!(array.ptype().is_signed_int());

        let err = BitPackedArray::encode(array.as_ref(), 1024u32.ilog2() as u8).unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }

    #[test]
    fn canonicalize_chunked_of_bitpacked() {
        let mut rng = StdRng::seed_from_u64(0);

        let chunks = (0..10)
            .map(|_| make_array(&mut rng, 100, 0.25, 0.25).unwrap())
            .collect::<Vec<_>>();
        let chunked = ChunkedArray::from_iter(chunks).into_array();

        let into_ca = chunked.clone().to_primitive();
        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        chunked.clone().append_to_builder(&mut primitive_builder);
        let ca_into = primitive_builder.finish();

        assert_eq!(
            into_ca.as_slice::<i32>(),
            ca_into.to_primitive().as_slice::<i32>()
        );

        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        primitive_builder.extend_from_array(&chunked);
        let ca_into = primitive_builder.finish();

        assert_eq!(
            into_ca.as_slice::<i32>(),
            ca_into.to_primitive().as_slice::<i32>()
        );
    }

    #[test]
    fn test_unpack_into_empty_array() {
        let empty: PrimitiveArray = PrimitiveArray::from_iter(Vec::<u32>::new());
        let bitpacked = bitpack_encode(&empty, 0, None).unwrap();

        let mut builder = PrimitiveBuilder::<u32>::new(Nullability::NonNullable);
        unpack_into(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();
        assert_eq!(
            result.len(),
            0,
            "Empty array should result in empty builder"
        );
    }

    /// This test ensures that the mask is properly appended to the range, not the builder.
    #[test]
    fn test_unpack_into_with_validity_mask() {
        // Create an array with some null values.
        let values = Buffer::from_iter([1u32, 0, 3, 4, 0]);
        let validity = Validity::from_iter([true, false, true, true, false]);
        let array = PrimitiveArray::new(values, validity);

        // Bitpack the array.
        let bitpacked = bitpack_encode(&array, 3, None).unwrap();

        // Unpack into a new builder.
        let mut builder = PrimitiveBuilder::<u32>::with_capacity(Nullability::Nullable, 5);
        unpack_into(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();

        // Verify the validity mask was correctly applied.
        assert_eq!(result.len(), 5);
        assert!(!result.scalar_at(0).is_null());
        assert!(result.scalar_at(1).is_null());
        assert!(!result.scalar_at(2).is_null());
        assert!(!result.scalar_at(3).is_null());
        assert!(result.scalar_at(4).is_null());
    }

    /// Test that `unpack_into` correctly handles arrays with patches.
    #[test]
    fn test_unpack_into_with_patches() {
        // Create an array where most values fit in 4 bits but some need patches.
        let values: Vec<u32> = (0..100)
            .map(|i| if i % 20 == 0 { 1000 + i } else { i % 16 })
            .collect();
        let array = PrimitiveArray::from_iter(values.clone());

        // Bitpack with a bit width that will require patches.
        let bitpacked = bitpack_encode(&array, 4, None).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Should have patches for values > 15"
        );

        // Unpack into a new builder.
        let mut builder = PrimitiveBuilder::<u32>::with_capacity(Nullability::NonNullable, 100);
        unpack_into(&bitpacked, &mut builder);

        let result = builder.finish_into_primitive();

        // Verify all values were correctly unpacked including patches.
        assert_eq!(result.as_slice::<u32>(), &values);
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
        assert_eq!(chunk_offsets.as_slice::<u64>(), &[0, 2, 2, 3]);
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

        assert_eq!(chunk_offsets.as_slice::<u64>(), &[0, 2, 2]);
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
        assert_eq!(chunk_offsets.as_slice::<u64>(), &[0, 2, 3, 3, 3]);
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
        assert_eq!(chunk_offsets.as_slice::<u64>(), &[0]);
    }
}
