// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::PrimInt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::BitPackedArray;
use crate::bitpack_decompress::count_exceptions;

/// The result of bit-packing an array.
#[derive(Debug)]
pub enum Packed {
    // TODO(aduffy): hold onto the stats?
    Unpatched(BitPackedArray),
    Patched(BitPackedArray, Patches),
}

impl Packed {
    pub fn has_patches(&self) -> bool {
        matches!(self, Self::Patched(_, _))
    }

    /// Unwrap the `packed` structure as the `Packed` variant without patches.
    ///
    /// # Panics
    ///
    /// Will panic if there are patches.
    pub fn unwrap_unpatched(self) -> BitPackedArray {
        match self {
            Self::Unpatched(unpacked) => unpacked,
            Self::Patched(..) => panic!("cannot unwrap Patched values as Unpatched"),
        }
    }

    /// Consume and retrieve only the packed result, discarding any patches.
    pub fn into_packed(self) -> BitPackedArray {
        match self {
            Packed::Unpatched(packed) => packed,
            Packed::Patched(packed, _) => packed,
        }
    }

    /// Get the full `ArrayRef` for the packed result.
    ///
    /// This will either point to a raw `BitPackedArray`, or a `PatchedArray` with a
    /// `BitPackedArray` child.
    ///
    /// # Errors
    ///
    /// If there are patches, we need to perform an array execution to transpose the patches. This
    /// will propagate any error from calling `execute` on the patches components.
    pub fn into_array(self) -> VortexResult<ArrayRef> {
        // We might need to execute the patches instead.
        match self {
            Packed::Unpatched(unpatched) => Ok(unpatched.into_array()),
            Packed::Patched(packed, patches) => Ok(PatchedArray::from_array_and_patches(
                packed.into_array(),
                &patches,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .into_array()),
        }
    }

    /// Apply a function to the patches, returning a new set of patches.
    pub fn map_patches<F, R>(self, func: F) -> VortexResult<Self>
    where
        F: FnOnce(Patches) -> VortexResult<Patches>,
    {
        match self {
            Packed::Unpatched(packed) => Ok(Packed::Unpatched(packed)),
            Packed::Patched(packed, patches) => {
                let mapped = func(patches)?;
                Ok(Packed::Patched(packed, mapped))
            }
        }
    }
}

pub struct BitPackEncoder<'a> {
    array: &'a PrimitiveArray,
    bit_width: Option<u8>,
    histogram: Option<&'a [usize]>,
}

impl<'a> BitPackEncoder<'a> {
    pub fn new(array: &'a PrimitiveArray) -> Self {
        Self {
            array,
            bit_width: None,
            histogram: None,
        }
    }

    /// Configure the encoder with a pre-selected bit-width for the output.
    ///
    /// If this is not configured, `pack` will scan the values and determine the optimal bit-width
    /// for compression.
    pub fn with_bit_width(mut self, bit_width: u8) -> Self {
        self.bit_width = Some(bit_width);
        self
    }

    /// Configure the encoder with a pre-computed histogram of values by bit-width.
    ///
    /// If not set, `pack` will scan the values and build the histogram.
    pub fn with_histogram(mut self, histogram: &'a [usize]) -> Self {
        self.histogram = Some(histogram);
        self
    }

    /// Consume the encoder and return the packed result. Any configured bit-width will be
    /// respected.
    ///
    /// # Error
    ///
    /// Packing will return an error if [`bitpack_encode`] would return an error, namely if the
    /// types or values of the input `PrimitiveArray` are out of range.
    pub fn pack(mut self) -> VortexResult<Packed> {
        let bit_width_freq = bit_width_histogram(self.array)?;
        let bw: u8 = match self.bit_width.take() {
            Some(bw) => bw,
            None => find_best_bit_width(self.array.ptype(), &bit_width_freq)?,
        };

        let (packed, patches) = bitpack_encode(self.array, bw, Some(&bit_width_freq))?;
        match patches {
            Some(patches) => Ok(Packed::Patched(packed, patches)),
            None => Ok(Packed::Unpatched(packed)),
        }
    }
}

/// Find the ideal bit width that maximally compresses the input array.
///
/// Returns the bit-packed, possibly patched, array.
pub fn bitpack_to_best_bit_width(array: &PrimitiveArray) -> VortexResult<ArrayRef> {
    BitPackEncoder::new(array).pack()?.into_array()
}

#[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
fn bitpack_encode(
    array: &PrimitiveArray,
    bit_width: u8,
    bit_width_freq: Option<&[usize]>,
) -> VortexResult<(BitPackedArray, Option<Patches>)> {
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
            vortex_bail!(InvalidArgument: "cannot bitpack_encode array containing negative integers")
        }
    }

    let num_exceptions = count_exceptions(bit_width, bit_width_freq);

    if bit_width >= array.ptype().bit_width() as u8 {
        // Nothing we can do
        vortex_bail!(
            InvalidArgument: "Cannot pack - specified bit width {bit_width} >= {}",
            array.ptype().bit_width()
        )
    }

    let packed = bitpack(array, bit_width)?;
    let patches = (num_exceptions > 0)
        .then(|| gather_patches(array, bit_width, num_exceptions))
        .transpose()?
        .flatten();

    // SAFETY: all components validated above
    let bitpacked = unsafe {
        BitPackedArray::new_unchecked(
            BufferHandle::new_host(packed),
            array.dtype().clone(),
            array.validity().clone(),
            bit_width,
            array.len(),
            0,
        )
    };

    // TODO(aduffy): I don't think this is correct. We should set stats on the outer PatchedArray
    //  instead maybe?
    //
    // bitpacked
    //     .stats_set
    //     .to_ref(bitpacked.as_ref())
    //     .inherit_from(array.statistics());
    Ok((bitpacked, patches))
}

/// Bitpack a [PrimitiveArray] to the given width.
///
/// On success, returns a [Buffer] containing the packed data.
fn bitpack(parray: &PrimitiveArray, bit_width: u8) -> VortexResult<ByteBuffer> {
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

pub fn bit_width_histogram(array: &PrimitiveArray) -> VortexResult<Vec<usize>> {
    match_each_integer_ptype!(array.ptype(), |P| { bit_width_histogram_typed::<P>(array) })
}

fn bit_width_histogram_typed<T: NativePType + PrimInt>(
    array: &PrimitiveArray,
) -> VortexResult<Vec<usize>> {
    let bit_width: fn(T) -> usize =
        |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);

    let mut bit_widths = vec![0usize; size_of::<T>() * 8 + 1];
    match array.validity_mask()?.bit_buffer() {
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

    use crate::bitpack_compress::BitPackEncoder;

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

        BitPackEncoder::new(&values)
            .with_bit_width(12)
            .pack()?
            .into_array()
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
    use crate::bitpack_compress::test_harness::make_array;

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
    fn compress_signed_fails() {
        let values: Buffer<i64> = (-500..500).collect();
        let array = PrimitiveArray::new(values, Validity::AllValid);
        assert!(array.ptype().is_signed_int());

        let err = BitPackEncoder::new(&array)
            .with_bit_width(10)
            .pack()
            .unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }

    #[test]
    fn canonicalize_chunked_of_bitpacked() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(0);

        let chunks = (0..10)
            .map(|_| make_array(&mut rng, 100, 0.25, 0.25).unwrap())
            .collect::<Vec<_>>();
        let chunked = ChunkedArray::from_iter(chunks).into_array();

        let into_ca = chunked.clone().to_primitive();
        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        chunked
            .clone()
            .append_to_builder(&mut primitive_builder, &mut SESSION.create_execution_ctx())?;
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into);

        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
        primitive_builder.extend_from_array(&chunked);
        let ca_into = primitive_builder.finish();

        assert_arrays_eq!(into_ca, ca_into);

        Ok(())
    }
}
