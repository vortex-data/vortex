// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::PrimInt;
use num_traits::Zero;
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
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
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

/// Bit-pack an array choosing the cost-model-optimal width for every 1024-element chunk.
///
/// Each chunk is charged for its packed block plus the exceptions left behind, so a chunk of small
/// values stays narrow no matter how wide its neighbours are.
///
/// Every chunk is processed in one go while it sits in L1: histogram, width choice, exception
/// gathering and packing, so the values are streamed from memory once.
pub fn bitpack_to_best_chunk_widths(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    ensure_non_negative(array, ctx)?;
    let validity = array.validity()?;
    let mask = validity.execute_mask(array.len(), ctx)?;
    let patch_validity = match validity {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let len = array.len();
    let (widths, packed, patches) = match_each_integer_ptype!(array.ptype(), |T| {
        encode_chunks::<T>(array.as_slice::<T>(), &mask, patch_validity)?
    });

    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        validity,
        patches,
        widths,
        len,
        0,
    )?;
    bitpacked.statistics().inherit_from(array.statistics());
    Ok(bitpacked)
}

/// Multi-pass reference for [`bitpack_to_best_chunk_widths`]: one pass to choose widths, one to
/// pack, and one to gather exceptions. Kept to check the fused encoder against.
#[cfg(test)]
pub(crate) fn bitpack_to_best_chunk_widths_multipass(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    let plan = chunk_width_plan(array.as_view(), ctx)?;
    bitpack_encode_planned(array, plan, ctx)
}

/// Histogram, choose a width, gather exceptions and pack, one 1024-element chunk at a time.
///
/// Patch indices use the narrowest unsigned type that can address every value.
fn encode_chunks<T>(
    values: &[T],
    mask: &Mask,
    patch_validity: Validity,
) -> VortexResult<(ChunkWidths, ByteBuffer, Option<Patches>)>
where
    T: NativePType + PrimInt + PhysicalPType,
    T::Physical: BitPacking + NativePType,
{
    let len = values.len();
    if len < u8::MAX as usize {
        encode_chunks_indexed::<T, u8>(values, mask, patch_validity)
    } else if len < u16::MAX as usize {
        encode_chunks_indexed::<T, u16>(values, mask, patch_validity)
    } else if len < u32::MAX as usize {
        encode_chunks_indexed::<T, u32>(values, mask, patch_validity)
    } else {
        encode_chunks_indexed::<T, u64>(values, mask, patch_validity)
    }
}

fn encode_chunks_indexed<T, P>(
    values: &[T],
    mask: &Mask,
    patch_validity: Validity,
) -> VortexResult<(ChunkWidths, ByteBuffer, Option<Patches>)>
where
    T: NativePType + PrimInt + PhysicalPType,
    T::Physical: BitPacking + NativePType,
    P: IntegerPType,
{
    let bits = T::PTYPE.bit_width();
    let bytes_per_exception = bytes_per_exception(T::PTYPE);
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);

    let mut widths = BufferMut::<u8>::with_capacity(num_chunks);
    let mut chunk_offsets = BufferMut::<u64>::with_capacity(num_chunks);
    let mut indices = BufferMut::<P>::empty();
    let mut patch_values = BufferMut::<T>::empty();

    let validity = match mask.bit_buffer() {
        AllOr::All => None,
        AllOr::Some(bits) => Some(bits),
        // Every value is null: every chunk is zero-width and there is nothing to pack.
        AllOr::None => {
            widths.extend_trusted(std::iter::repeat_n(0u8, num_chunks));
            chunk_offsets.extend_trusted(std::iter::repeat_n(0u64, num_chunks));
            return Ok((ChunkWidths::new(widths.freeze()), ByteBuffer::empty(), None));
        }
    };

    // Every chunk packs into a whole block, so the padded size bounds the output; a short trailing
    // chunk can pack to more than its raw size. The buffer is shrunk to its exact size at the end.
    let mut packed = BufferMut::<T::Physical>::with_capacity(num_chunks * FL_CHUNK_SIZE);
    let mut histogram = vec![0usize; bits + 1];
    // Zero-padded copy of the trailing partial chunk.
    let mut padded = [T::Physical::zero(); FL_CHUNK_SIZE];

    for (chunk_idx, chunk) in values.chunks(FL_CHUNK_SIZE).enumerate() {
        let base = chunk_idx * FL_CHUNK_SIZE;
        let chunk_validity = validity.map(|v| v.slice(base..base + chunk.len()));

        histogram.fill(0);
        for_each_valid_width(chunk, chunk_validity.as_ref(), |_, _, width| {
            histogram[width] += 1;
        });

        let bit_width = best_chunk_width(&histogram, bytes_per_exception);
        widths.push(bit_width);
        chunk_offsets.push(patch_values.len() as u64);

        // The chunk is still in L1, so a second walk over it is cheaper than remembering widths.
        if count_exceptions(bit_width, &histogram) > 0 {
            for_each_valid_width(chunk, chunk_validity.as_ref(), |i, value, width| {
                if width > bit_width as usize {
                    indices.push(P::from(base + i).vortex_expect("cast index from usize"));
                    patch_values.push(value);
                }
            });
        }

        if bit_width > 0 {
            let input: &[T::Physical] = if chunk.len() == FL_CHUNK_SIZE {
                as_physical(chunk)
            } else {
                padded[..chunk.len()].copy_from_slice(as_physical(chunk));
                &padded
            };
            let packed_len = chunk_packed_bytes(bit_width) / size_of::<T::Physical>();
            let start = packed.len();
            // SAFETY: `input` holds exactly 1024 values and the output window is exactly one
            // packed block at `bit_width`, within the raw-size capacity reserved above.
            unsafe {
                packed.set_len(start + packed_len);
                BitPacking::unchecked_pack(bit_width as usize, input, &mut packed[start..]);
            }
        }
    }

    let packed = if packed.len() < packed.capacity() {
        let mut exact = BufferMut::<T::Physical>::with_capacity(packed.len());
        exact.extend_from_slice(&packed);
        exact.freeze()
    } else {
        packed.freeze()
    };

    let patches = if indices.is_empty() {
        None
    } else {
        Some(Patches::new(
            values.len(),
            0,
            indices.into_array(),
            PrimitiveArray::new(patch_values, patch_validity).into_array(),
            Some(chunk_offsets.into_array()),
        )?)
    };

    Ok((
        ChunkWidths::new(widths.freeze()),
        packed.into_byte_buffer(),
        patches,
    ))
}

/// Call `f(index, value, bit_width)` for every value of `chunk`; nulls report a width of zero.
#[inline]
fn for_each_valid_width<T: NativePType + PrimInt>(
    chunk: &[T],
    validity: Option<&BitBuffer>,
    mut f: impl FnMut(usize, T, usize),
) {
    let bits = T::PTYPE.bit_width();
    match validity {
        None => {
            for (i, &v) in chunk.iter().enumerate() {
                f(i, v, bits - PrimInt::leading_zeros(v) as usize);
            }
        }
        Some(validity) => {
            for ((i, &v), valid) in chunk.iter().enumerate().zip(validity.iter()) {
                let width = if valid {
                    bits - PrimInt::leading_zeros(v) as usize
                } else {
                    0
                };
                f(i, v, width);
            }
        }
    }
}

/// View signed or unsigned values as their unsigned physical twin, which FastLanes packs.
fn as_physical<T: PhysicalPType>(values: &[T]) -> &[T::Physical] {
    const {
        assert!(size_of::<T>() == size_of::<T::Physical>());
        assert!(align_of::<T>() == align_of::<T::Physical>());
    }
    // SAFETY: `Physical` is the same-width unsigned integer, so the layouts match exactly.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

/// The cost-model-optimal bit width of every 1024-element chunk of `array`.
pub fn best_chunk_widths(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ChunkWidths> {
    Ok(chunk_width_plan(array, ctx)?.widths)
}

/// Bit-pack `array` at the given per-chunk widths, gathering values that do not fit their chunk's
/// width into patches.
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
/// `fastlanes.bitpacked` format. See [`bitpack_to_best_chunk_widths`] for per-chunk widths.
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
    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        None,
        widths,
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

fn chunk_width_plan(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ChunkWidthPlan> {
    match_each_integer_ptype!(array.ptype(), |P| {
        chunk_width_plan_typed::<P>(array, ctx)
    })
}

fn chunk_width_plan_typed<T: NativePType + PrimInt>(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ChunkWidthPlan> {
    let bytes_per_exception = bytes_per_exception(T::PTYPE);
    let values = array.as_slice::<T>();
    let num_chunks = values.len().div_ceil(FL_CHUNK_SIZE);
    let bit_width: fn(T) -> usize =
        |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);

    let mut widths = BufferMut::<u8>::with_capacity(num_chunks);
    let mut num_exceptions = 0usize;
    let mut histogram = vec![0usize; size_of::<T>() * 8 + 1];

    // Score one chunk's histogram and reset it for the next chunk.
    let mut finish_chunk = |histogram: &mut [usize]| -> u8 {
        let best = best_chunk_width(histogram, bytes_per_exception);
        num_exceptions += count_exceptions(best, histogram);
        histogram.fill(0);
        best
    };

    match array
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
        .bit_buffer()
    {
        AllOr::All => {
            for chunk in values.chunks(FL_CHUNK_SIZE) {
                for v in chunk {
                    histogram[bit_width(*v)] += 1;
                }
                widths.push(finish_chunk(&mut histogram));
            }
        }
        AllOr::None => {
            for _ in 0..num_chunks {
                widths.push(0);
            }
        }
        AllOr::Some(buffer) => {
            let mut valid = buffer.iter();
            for chunk in values.chunks(FL_CHUNK_SIZE) {
                for v in chunk {
                    if valid.next().unwrap_or(true) {
                        histogram[bit_width(*v)] += 1;
                    } else {
                        histogram[0] += 1;
                    }
                }
                widths.push(finish_chunk(&mut histogram));
            }
        }
    }

    Ok(ChunkWidthPlan {
        widths: ChunkWidths::new(widths.freeze()),
        num_exceptions: Some(num_exceptions),
    })
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

    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(packed),
        array.ptype(),
        array.validity()?,
        patches,
        widths,
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

/// The width minimising one chunk's cost: its packed block plus the exceptions left behind.
///
/// A chunk always occupies a whole `128 * width` byte block, so a partial trailing chunk is
/// charged for its padding.
fn best_chunk_width(bit_width_freq: &[usize], bytes_per_exception: usize) -> u8 {
    let len: usize = bit_width_freq.iter().sum();
    let mut num_packed = 0;
    let mut best_cost = usize::MAX;
    let mut best_width = 0;
    for (bit_width, freq) in bit_width_freq.iter().enumerate() {
        num_packed += *freq;
        let cost = chunk_packed_bytes(bit_width as u8) + (len - num_packed) * bytes_per_exception;
        if cost < best_cost {
            best_cost = cost;
            best_width = bit_width;
        }
    }
    best_width as u8
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
    fn test_best_chunk_width() {
        // 1000 3-bit values and 24 10-bit values in a u16 chunk: 3 bits plus 24 exceptions
        // (384 + 24 * 6 bytes) beats 10 bits for everything (1280 bytes).
        let mut freq = vec![0usize; 17];
        freq[3] = 1000;
        freq[10] = 24;
        assert_eq!(best_chunk_width(&freq, bytes_per_exception(PType::U16)), 3);
        // Make the exceptions expensive enough and the wide width wins.
        freq[10] = 200;
        assert_eq!(best_chunk_width(&freq, bytes_per_exception(PType::U16)), 10);
    }

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
        let mut primitive_builder =
            PrimitiveBuilder::<i32>::with_capacity(chunked.dtype().nullability(), 10 * 100);
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
