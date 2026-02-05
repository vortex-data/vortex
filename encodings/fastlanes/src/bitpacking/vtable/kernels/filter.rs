// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::sync::Arc;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::FilterKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::filter::Filter;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::UnsignedPType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::BitPackedArray;
use crate::BitPackedVTable;
use crate::bitpacking::vtable::kernels::UNPACK_CHUNK_THRESHOLD;
use crate::bitpacking::vtable::kernels::chunked_indices;

pub(crate) const PARENT_KERNELS: ParentKernelSet<BitPackedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(BitPackedVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(BitPackedVTable)),
]);

/// The threshold over which it is faster to fully unpack the entire [`BitPackedArray`] and then
/// filter the result than to unpack only specific bitpacked values into the output buffer.
pub const fn unpack_then_filter_threshold(ptype: PType) -> f64 {
    // TODO(connor): Where did these numbers come from? Add a public link after validating them.
    // These numbers probably don't work for in-place filtering either.
    match ptype.byte_width() {
        1 => 0.03,
        2 => 0.03,
        4 => 0.075,
        _ => 0.09,
        // >8 bytes may have a higher threshold. These numbers are derived from a GCP c2-standard-4
        // with a "Cascade Lake" CPU.
    }
}

/// Kernel to execute filtering directly on a bit-packed array.
impl FilterKernel for BitPackedVTable {
    fn filter(
        array: &BitPackedArray,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => {
                return Ok(None);
            }
            Mask::Values(values) => values,
        };

        // If the density is high enough, then we would rather decompress the whole array and then apply
        // a filter over decompressing values one by one.
        if values.density() > unpack_then_filter_threshold(array.ptype()) {
            return Ok(None);
        }

        // Filter and patch using the correct unsigned type for FastLanes, then cast to signed if needed.
        let mut primitive = match_each_unsigned_integer_ptype!(array.ptype().to_unsigned(), |U| {
            let (buffer, validity) = filter_primitive_without_patches::<U>(array, values)?;

            let validity = Validity::from_mask(validity, array.dtype().nullability());
            // reinterpret_cast for signed types.
            PrimitiveArray::new(buffer, validity).reinterpret_cast(array.ptype())
        });

        let patches = array
            .patches()
            .map(|patches| patches.filter(&Mask::Values(values.clone())))
            .transpose()?
            .flatten();

        if let Some(patches) = patches {
            primitive = primitive.patch(&patches)?;
        }

        Ok(Some(primitive.into_array()))
    }
}

/// Specialized filter kernel for primitive bit-packed arrays.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, the provided
/// `U` should be promoted to the unsigned variant for any target bit width.
/// For example, if the array is bit-packed `i16`, this function should be called with `U = u16`.
///
/// This function fully decompresses the array for all but the most selective masks because the
/// FastLanes decompression is so fast and the bookkeepping necessary to decompress individual
/// elements is relatively slow.
///
/// Returns a tuple of (values buffer, validity mask).
fn filter_primitive_without_patches<U: UnsignedPType + BitPacking>(
    array: &BitPackedArray,
    selection: &Arc<MaskValues>,
) -> VortexResult<(Buffer<U>, Mask)> {
    let values = filter_with_indices(array, selection.indices());
    let validity = array
        .validity_mask()?
        .filter(&Mask::Values(selection.clone()))
        .into_mut();

    debug_assert_eq!(
        values.len(),
        validity.len(),
        "`filter_with_indices` was somehow incorrect"
    );

    Ok((values.freeze(), validity.freeze()))
}

fn filter_with_indices<T: NativePType + BitPacking>(
    array: &BitPackedArray,
    indices: &[usize],
) -> BufferMut<T> {
    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;
    let mut values = BufferMut::with_capacity(indices.len());

    // Some re-usable memory to store per-chunk indices.
    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; 1024];
    let packed_bytes = array.packed_slice::<T>();

    // Group the indices by the FastLanes chunk they belong to.
    let chunk_size = 128 * bit_width / size_of::<T>();

    chunked_indices(indices, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed_bytes[chunk_idx * chunk_size..][..chunk_size];

        if indices_within_chunk.len() == 1024 {
            // Unpack the entire chunk.
            unsafe {
                let values_len = values.len();
                values.set_len(values_len + 1024);
                BitPacking::unchecked_unpack(
                    bit_width,
                    packed,
                    &mut values.as_mut_slice()[values_len..],
                );
            }
        } else if indices_within_chunk.len() > UNPACK_CHUNK_THRESHOLD {
            // Unpack into a temporary chunk and then copy the values.
            unsafe {
                let dst: &mut [MaybeUninit<T>] = &mut unpacked;
                let dst: &mut [T] = std::mem::transmute(dst);
                BitPacking::unchecked_unpack(bit_width, packed, dst);
            }
            values.extend_trusted(
                indices_within_chunk
                    .iter()
                    .map(|&idx| unsafe { unpacked.get_unchecked(idx).assume_init() }),
            );
        } else {
            // Otherwise, unpack each element individually.
            values.extend_trusted(indices_within_chunk.iter().map(|&idx| unsafe {
                BitPacking::unchecked_unpack_single(bit_width, packed, idx)
            }));
        }
    });

    values
}
