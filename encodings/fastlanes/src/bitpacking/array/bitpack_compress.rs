// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::PrimInt;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
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
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_decompress::count_exceptions;
use crate::bitpacking::array::ChunkWidths;
use crate::bitpacking::array::chunk_packed_bytes;

/// Encode with caller-supplied chunk widths, gathering exceptions for values that do not fit.
pub fn bitpack_encode_with_widths(
    array: &PrimitiveArray,
    widths: ChunkWidths,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    let num_chunks = array.len().div_ceil(FL_CHUNK_SIZE);
    vortex_ensure!(
        widths.len() == num_chunks,
        "Expected {num_chunks} chunk widths for {} values, got {}",
        array.len(),
        widths.len()
    );
    let plan = ChunkWidthPlan {
        widths,
        num_exceptions: None,
    };
    bitpack_encode_planned(array, plan, ctx)
}

/// Bit-pack `array` at the single best global width chosen by [`find_best_bit_width`].
///
/// Every chunk shares that width, so the result serializes under the original
/// `fastlanes.bitpacked` format.
pub fn bitpack_to_best_bit_width(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    let bit_width_freq = bit_width_histogram(array.as_view(), ctx)?;
    let best_bit_width = find_best_bit_width(array.ptype(), &bit_width_freq)?;
    bitpack_encode(array, best_bit_width, Some(&bit_width_freq), ctx)
}

/// Bit-pack every chunk of `array` at the same `bit_width`, which must be narrower than the type.
///
/// `bit_width_freq` is the array's bit-width histogram if already known; it saves recomputing it
/// to count exceptions.
pub fn bitpack_encode(
    array: &PrimitiveArray,
    bit_width: u8,
    bit_width_freq: Option<&[usize]>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    if bit_width as usize >= array.ptype().bit_width() {
        vortex_bail!(
            InvalidArgument: "Cannot pack - specified bit width {bit_width} >= {}",
            array.ptype().bit_width()
        )
    }
    let num_exceptions = bit_width_freq.map(|freq| count_exceptions(bit_width, freq));
    let widths = ChunkWidths::uniform(bit_width, array.len().div_ceil(FL_CHUNK_SIZE));
    bitpack_encode_planned(
        array,
        ChunkWidthPlan {
            widths,
            num_exceptions,
        },
        ctx,
    )
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
    let widths = ChunkWidths::uniform(bit_width, array.len().div_ceil(FL_CHUNK_SIZE));
    // SAFETY: non-negativity of input checked by caller.
    let packed = unsafe { bitpack_unchecked_with_widths(&array, &widths) };

    let arr_ref = array.clone().into_array();
    let offsets = widths.offsets_array();
    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        None,
        widths.into_array(),
        offsets,
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
    let widths = ChunkWidths::uniform(bit_width, parray.len().div_ceil(FL_CHUNK_SIZE));
    // SAFETY: forwarded to the caller.
    unsafe { bitpack_unchecked_with_widths(parray, &widths) }
}

/// Bitpack a slice of primitives down to the given width.
///
/// See `bitpack` for more caller information.
pub fn bitpack_primitive<T: NativePType + BitPacking>(array: &[T], bit_width: u8) -> Buffer<T> {
    let widths = ChunkWidths::uniform(bit_width, array.len().div_ceil(FL_CHUNK_SIZE));
    bitpack_primitive_chunked(array, &widths)
}

/// Gather the values that do not fit `bit_width` into patches.
pub fn gather_patches(
    parray: &PrimitiveArray,
    bit_width: u8,
    num_exceptions_hint: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Patches>> {
    let widths = ChunkWidths::uniform(bit_width, parray.len().div_ceil(FL_CHUNK_SIZE));
    gather_patches_with_widths(parray, &widths, num_exceptions_hint, ctx)
}

/// Chosen chunk widths plus, when known, how many values do not fit them.
struct ChunkWidthPlan {
    widths: ChunkWidths,
    num_exceptions: Option<usize>,
}

fn bitpack_encode_planned(
    array: &PrimitiveArray,
    plan: ChunkWidthPlan,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    ensure_non_negative(array, ctx)?;
    let ChunkWidthPlan {
        widths,
        num_exceptions,
    } = plan;

    // SAFETY: we check that array only contains non-negative values.
    let packed = unsafe { bitpack_unchecked_with_widths(array, &widths) };
    let patches = if num_exceptions == Some(0) {
        None
    } else {
        gather_patches_with_widths(array, &widths, num_exceptions.unwrap_or(0), ctx)?
    };

    let offsets = widths.offsets_array();
    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        patches,
        widths.into_array(),
        offsets,
        array.len(),
        0,
    )?;
    bitpacked.statistics().inherit_from(array.statistics());
    Ok(bitpacked)
}

#[expect(unused_comparisons, clippy::absurd_extreme_comparisons)]
fn ensure_non_negative(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    if array.ptype().is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(array.ptype(), |P| {
            array.statistics().compute_min::<P>(ctx).unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!(InvalidArgument: "cannot bitpack_encode array containing negative integers")
        }
    }
    Ok(())
}

/// Bitpack a [PrimitiveArray] with one width per 1024-element chunk.
///
/// # Safety
///
/// Internally this function will promote the provided array to its unsigned equivalent. This will
/// violate ordering guarantees if the array contains any negative values, so the caller must
/// ensure that `parray` is non-negative.
pub unsafe fn bitpack_unchecked_with_widths(
    parray: &PrimitiveArray,
    widths: &ChunkWidths,
) -> ByteBuffer {
    let parray = parray.reinterpret_cast(parray.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(parray.ptype(), |P| {
        bitpack_primitive_chunked(parray.as_slice::<P>(), widths).into_byte_buffer()
    })
}

/// Bitpack a slice of primitives, packing each 1024-element chunk at its own width.
///
/// Chunks of width zero contribute no packed bytes; the trailing partial chunk is zero-padded.
pub fn bitpack_primitive_chunked<T: NativePType + BitPacking>(
    array: &[T],
    widths: &ChunkWidths,
) -> Buffer<T> {
    let mut output = BufferMut::<T>::with_capacity(widths.packed_bytes() / size_of::<T>());
    let mut last_chunk = [T::zero(); FL_CHUNK_SIZE];

    for (chunk_idx, chunk) in array.chunks(FL_CHUNK_SIZE).enumerate() {
        let bit_width = widths.width(chunk_idx);
        if bit_width == 0 {
            continue;
        }
        let packed_len = chunk_packed_bytes(bit_width) / size_of::<T>();
        let input: &[T] = if chunk.len() == FL_CHUNK_SIZE {
            chunk
        } else {
            last_chunk[..chunk.len()].copy_from_slice(chunk);
            &last_chunk
        };

        let output_len = output.len();
        // SAFETY: `input` holds exactly 1024 values and the output window is exactly one packed
        // block at `bit_width`, which the capacity reserved above accounts for.
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width as usize,
                input,
                &mut output[output_len..][..packed_len],
            );
        }
    }

    output.freeze()
}

/// Gather the values that do not fit their chunk's bit width into patches.
pub fn gather_patches_with_widths(
    parray: &PrimitiveArray,
    widths: &ChunkWidths,
    num_exceptions_hint: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Patches>> {
    let patch_validity = match parray.validity()? {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let array_len = parray.len();
    let validity_mask = parray
        .as_ref()
        .validity()?
        .execute_mask(parray.len(), ctx)?;

    let patches = if array_len < u8::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u8>(
                parray.as_slice::<T>(),
                widths,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u16::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u16>(
                parray.as_slice::<T>(),
                widths,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u32::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u32>(
                parray.as_slice::<T>(),
                widths,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u64>(
                parray.as_slice::<T>(),
                widths,
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
    widths: &ChunkWidths,
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

    let total_chunks = data.len().div_ceil(FL_CHUNK_SIZE);
    let mut chunk_offsets: BufferMut<u64> = BufferMut::with_capacity(total_chunks);

    // A value overflows its chunk's width when it has fewer leading zeros than this.
    let mut overflow_leading_zeros = 0usize;
    for ((idx, value), valid) in data.iter().enumerate().zip(validity_mask.iter()) {
        if idx.is_multiple_of(FL_CHUNK_SIZE) {
            // Record the patch index offset for each chunk.
            chunk_offsets.push(values.len() as u64);
            overflow_leading_zeros =
                T::PTYPE.bit_width() - widths.width(idx / FL_CHUNK_SIZE) as usize;
        }

        if (value.leading_zeros() as usize) < overflow_leading_zeros && valid {
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

pub fn bit_width_histogram(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<usize>> {
    match_each_integer_ptype!(array.ptype(), |P| {
        bit_width_histogram_typed::<P>(array, ctx)
    })
}

fn bit_width_histogram_typed<T: NativePType + PrimInt>(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<usize>> {
    let bit_width: fn(T) -> usize =
        |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);

    let mut bit_widths = vec![0usize; size_of::<T>() * 8 + 1];
    match array
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
        .bit_buffer()
    {
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

/// Exceptions cost their value plus a u32 index; we cannot predict how patches compress.
fn bytes_per_exception(ptype: PType) -> usize {
    ptype.byte_width() + 4
}

#[cfg(feature = "_test-harness")]
pub mod test_harness {
    use rand::RngExt;
    use rand::rngs::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
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
        ctx: &mut ExecutionCtx,
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
            values.into_array().execute::<PrimitiveArray>(ctx)?
        } else {
            let validity = Validity::from_iter((0..len).map(|_| !rng.random_bool(fraction_null)));
            PrimitiveArray::new(values, validity)
        };

        bitpack_encode(&values, 12, None, ctx).map(|a| a.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::PrimitiveBuilder;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

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

    /// Values below 100 with every 40th value pushed above 12 bits, and every 5th null.
    fn patchy_nullable(len: usize, seed: u32) -> PrimitiveArray {
        let values = (0..len as u32)
            .map(|i| {
                let v = (i * 7919 + seed) % 100;
                if i % 40 == 0 { v + (1 << 13) } else { v }
            })
            .map(|v| v as i32)
            .collect::<Buffer<i32>>();
        let validity = Validity::from_iter((0..len).map(|i| i % 5 != 0));
        PrimitiveArray::new(values, validity)
    }

    #[test]
    fn canonicalize_chunked_of_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let chunks = (0..10)
            .map(|seed| {
                bitpack_encode(&patchy_nullable(100, seed), 12, None, &mut ctx)
                    .map(|a| a.into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
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
        Ok(())
    }

    fn chunk_offsets_of(values: Vec<u32>) -> VortexResult<PrimitiveArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter(values);
        let bitpacked = bitpack_encode(&array, 4, None, &mut ctx)?;
        let patches = bitpacked
            .patches()
            .ok_or_else(|| vortex_err!("expected patches"))?;
        patches
            .chunk_offsets()
            .as_ref()
            .ok_or_else(|| vortex_err!("expected chunk offsets"))?
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
    }

    fn with_patches(len: usize, patch_indices: &[usize]) -> Vec<u32> {
        let mut values = vec![0u32; len];
        patch_indices.iter().for_each(|&idx| values[idx] = 1 << 20);
        values
    }

    #[test]
    fn test_chunk_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // chunk 0: patches at 100, 200; chunk 1: none; chunk 2: 3000; chunk 3: 3100
        assert_arrays_eq!(
            chunk_offsets_of(with_patches(4096, &[100, 200, 3000, 3100]))?,
            PrimitiveArray::from_iter([0u64, 2, 2, 3]),
            &mut ctx
        );
        // Trailing chunks without patches all point past the last patch.
        assert_arrays_eq!(
            chunk_offsets_of(with_patches(5120, &[100, 200, 1500]))?,
            PrimitiveArray::from_iter([0u64, 2, 3, 3, 3]),
            &mut ctx
        );
        assert_arrays_eq!(
            chunk_offsets_of(with_patches(500, &[100, 200]))?,
            PrimitiveArray::from_iter([0u64]),
            &mut ctx
        );
        Ok(())
    }
}
