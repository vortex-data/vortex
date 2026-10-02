// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod blocked;
mod global;

pub use blocked::bitpack_encode_blocked;
pub use blocked::bitpack_to_best_bit_widths;
use fastlanes::BitPacking;
pub use global::bitpack_encode;
pub use global::bitpack_encode_unchecked;
pub use global::bitpack_primitive;
pub use global::bitpack_to_best_bit_width;
pub use global::bitpack_unchecked;
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
use crate::BitWidths;
use crate::FL_CHUNK_SIZE;

/// Return an error unless `array` holds integers that are all non-negative, which bit-packing
/// requires.
#[expect(unused_comparisons, clippy::absurd_extreme_comparisons)]
fn ensure_non_negative_integers(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    vortex_ensure!(
        array.ptype().is_int(),
        InvalidArgument: "cannot bitpack {} array",
        array.ptype()
    );
    if array.ptype().is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(array.ptype(), |P| {
            array.statistics().compute_min::<P>(ctx).unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!(InvalidArgument: "cannot bitpack array containing negative integers")
        }
    }
    Ok(())
}

/// Bitpack each 1024-value block of `parray` at `block_width(block)` bits.
///
/// # Safety
///
/// This promotes `parray` to its unsigned equivalent, like [`bitpack_unchecked`], so the caller
/// must ensure that it holds no negative values.
unsafe fn pack_blocks_unchecked(
    parray: &PrimitiveArray,
    block_width: &dyn Fn(usize) -> u8,
) -> ByteBuffer {
    let parray = parray.reinterpret_cast(parray.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(parray.ptype(), |P| {
        pack_blocks(parray.as_slice::<P>(), block_width).into_byte_buffer()
    })
}

/// Bitpack each 1024-value block of `array` at `block_width(block)` bits, one block after another.
fn pack_blocks<T: NativePType + BitPacking>(
    array: &[T],
    block_width: &dyn Fn(usize) -> u8,
) -> Buffer<T> {
    let block_len = |block: usize| 128 * usize::from(block_width(block)) / size_of::<T>();
    let num_blocks = array.len().div_ceil(FL_CHUNK_SIZE);
    let mut output = BufferMut::<T>::with_capacity((0..num_blocks).map(block_len).sum());
    let mut pack_block = |block_idx: usize, input: &[T]| {
        let width = usize::from(block_width(block_idx));
        let len = 128 * width / size_of::<T>();
        let output_len = output.len();
        // SAFETY: `input` holds 1024 values and the output window is exactly one block packed at
        // its width, within the capacity reserved above.
        unsafe {
            output.set_len(output_len + len);
            BitPacking::unchecked_pack(width, input, &mut output[output_len..][..len]);
        }
    };

    let mut blocks = array.chunks_exact(FL_CHUNK_SIZE);
    for (block_idx, block) in blocks.by_ref().enumerate() {
        pack_block(block_idx, block);
    }
    // Only a partial last block is zero-padded, so that the zeroing stays off the common path.
    let remainder = blocks.remainder();
    if !remainder.is_empty() {
        let mut padded = [T::zero(); FL_CHUNK_SIZE];
        padded[..remainder.len()].copy_from_slice(remainder);
        pack_block(num_blocks - 1, &padded);
    }

    output.freeze()
}

/// Assemble a [`BitPackedArray`] holding `array`'s packed values, validity and statistics.
fn bitpacked_from_packed(
    array: &PrimitiveArray,
    packed: ByteBuffer,
    patches: Option<Patches>,
    bit_widths: BitWidths,
) -> VortexResult<BitPackedArray> {
    let packed = BufferHandle::new_host(packed);
    let validity = array.validity()?;
    let bitpacked = match bit_widths {
        BitWidths::Global(bit_width) => BitPacked::try_new(
            packed,
            array.ptype(),
            validity,
            patches,
            bit_width,
            array.len(),
            0,
        )?,
        BitWidths::Blocked(block_offsets) => BitPacked::try_new_with_block_offsets(
            packed,
            array.ptype(),
            validity,
            patches,
            block_offsets,
            array.len(),
            0,
        )?,
    };
    bitpacked.statistics().inherit_from(array.statistics());
    Ok(bitpacked)
}

pub fn gather_patches(
    parray: &PrimitiveArray,
    bit_width: u8,
    num_exceptions_hint: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Patches>> {
    let validity_mask = parray
        .as_ref()
        .validity()?
        .execute_mask(parray.len(), ctx)?;
    gather_patches_with(parray, &|_| bit_width, num_exceptions_hint, validity_mask)
}

/// Gather the valid values that are wider than the bit width of their 1024-value block.
fn gather_patches_with(
    parray: &PrimitiveArray,
    block_width: &dyn Fn(usize) -> u8,
    num_exceptions_hint: usize,
    validity_mask: Mask,
) -> VortexResult<Option<Patches>> {
    let patch_validity = match parray.validity()? {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let array_len = parray.len();

    let patches = if array_len < u8::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u8>(
                parray.as_slice::<T>(),
                block_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u16::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u16>(
                parray.as_slice::<T>(),
                block_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else if array_len < u32::MAX as usize {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u32>(
                parray.as_slice::<T>(),
                block_width,
                num_exceptions_hint,
                patch_validity,
                validity_mask,
            )?
        })
    } else {
        match_each_integer_ptype!(parray.ptype(), |T| {
            gather_patches_impl::<T, u64>(
                parray.as_slice::<T>(),
                block_width,
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
    block_width: &dyn Fn(usize) -> u8,
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

    let mut bit_width = 0;
    for ((idx, value), valid) in data.iter().enumerate().zip(validity_mask.iter()) {
        if (idx % 1024) == 0 {
            // Record the patch index offset and bit width for each chunk.
            chunk_offsets.push(values.len() as u64);
            bit_width = block_width(idx / 1024);
        }

        if (value.leading_zeros() as usize) < T::PTYPE.bit_width() - bit_width as usize && valid {
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
    let mut bit_widths = vec![0usize; size_of::<T>() * 8 + 1];
    let validity = array.validity()?.execute_mask(array.as_ref().len(), ctx)?;
    add_bit_widths(
        &mut bit_widths,
        array.as_slice::<T>(),
        validity.bit_buffer(),
    );
    Ok(bit_widths)
}

/// Count the bit width of each of `values` into `histogram`, counting null values as zero-width.
fn add_bit_widths<T: NativePType + PrimInt>(
    histogram: &mut [usize],
    values: &[T],
    validity: AllOr<&BitBuffer>,
) {
    let bit_width: fn(T) -> usize =
        |v: T| (8 * size_of::<T>()) - (PrimInt::leading_zeros(v) as usize);
    match validity {
        AllOr::All => {
            for v in values {
                histogram[bit_width(*v)] += 1;
            }
        }
        AllOr::None => histogram[0] += values.len(),
        AllOr::Some(buffer) => {
            for (is_valid, v) in buffer.iter().zip_eq(values) {
                histogram[if is_valid { bit_width(*v) } else { 0 }] += 1;
            }
        }
    }
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
    use super::*;

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
}
