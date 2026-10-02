// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod blocked;
mod global;

pub use blocked::bitpack_encode_blocked;
pub use blocked::bitpack_to_best_bit_widths;
pub use global::bit_width_histogram;
pub use global::bitpack_encode;
pub use global::bitpack_encode_unchecked;
pub use global::bitpack_primitive;
pub use global::bitpack_to_best_bit_width;
pub use global::bitpack_unchecked;
pub use global::find_best_bit_width;
pub use global::gather_patches;
#[cfg(feature = "_test-harness")]
pub use global::test_harness;
use fastlanes::BitPacking;
use num_traits::PrimInt;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

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

/// Bitpack each 1024-value block of `array` at `bit_width(block)` bits, one block after another,
/// into a buffer with room for `capacity` values.
fn bitpack_blocks<T: NativePType + BitPacking>(
    array: &[T],
    capacity: usize,
    bit_width: impl Fn(usize) -> usize,
) -> Buffer<T> {
    // How many fastlanes vectors we will process.
    let num_chunks = array.len().div_ceil(1024);
    let num_full_chunks = array.len() / 1024;
    // packed_len says how many values of size T we're going to include.
    // 1024 * bit_width / 8 == the number of bytes we're going to get.
    // then we divide by the size of T to get the number of elements.
    let packed_len = |chunk: usize| 128 * bit_width(chunk) / size_of::<T>();

    // Allocate a result byte array.
    let mut output = BufferMut::<T>::with_capacity(capacity);

    // Loop over all but the last chunk.
    (0..num_full_chunks).for_each(|i| {
        let start_elem = i * 1024;
        let output_len = output.len();
        let packed_len = packed_len(i);
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width(i),
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
        let packed_len = packed_len(num_full_chunks);
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width(num_full_chunks),
                &last_chunk,
                &mut output[output_len..][..packed_len],
            );
        };
    }

    output.freeze()
}

/// Gather the valid values of `parray` that are wider than `bit_width(block)`, the bit width of
/// their 1024-value block.
fn gather_patches_with(
    parray: &PrimitiveArray,
    bit_width: impl Fn(usize) -> u8,
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
    bit_width: impl Fn(usize) -> u8,
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

    for ((idx, value), valid) in data.iter().enumerate().zip(validity_mask.iter()) {
        if (idx % 1024) == 0 {
            // Record the patch index offset for each chunk.
            chunk_offsets.push(values.len() as u64);
        }

        if (value.leading_zeros() as usize) < T::PTYPE.bit_width() - bit_width(idx / 1024) as usize
            && valid
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
