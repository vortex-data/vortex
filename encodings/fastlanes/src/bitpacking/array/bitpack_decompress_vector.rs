// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use vortex_buffer::BufferMut;
use vortex_compute::filter::Filter;
use vortex_dtype::match_each_integer_ptype;
use vortex_mask::Mask;
use vortex_vector::VectorMutOps;
use vortex_vector::primitive::{PVector, PVectorMut, PrimitiveVector};

use crate::BitPackedArray;
use crate::bitpacking::array::BitPacked;

/// The threshold over which it is faster to fully unpack the entire [`BitPackedArray`] and then
/// filter the result than to unpack only specific bitpacked values into the output buffer.
const fn unpack_then_filter_threshold<T>() -> f64 {
    // TODO(connor): Where did these numbers come from? Add a public link after validating them.
    // These numbers probably don't work for in-place filtering either.
    match size_of::<T>() {
        1 => 0.03,
        2 => 0.03,
        4 => 0.075,
        _ => 0.09,
        // >8 bytes may have a higher threshold. These numbers are derived from a GCP c2-standard-4
        // with a "Cascade Lake" CPU.
    }
}

/// Helper function to get the true count of a mask with a default if it doesn't exist.
fn true_count_of_optional_mask(selection_mask: &Option<&Mask>, default: usize) -> usize {
    selection_mask
        .map(|mask| mask.true_count())
        .unwrap_or(default)
}

/// Given a [`BitPackedArray`], unpacks all bitpacked values and creates a new [`PrimitiveVector`].
///
/// If a selection mask is passed in, the resultant vector will have the specified values filtered
/// out.
///
/// Internally, this calls [`unpack_into_pvector`].
pub fn unpack_into_vector(
    array: &BitPackedArray,
    selection_mask: Option<&Mask>,
) -> PrimitiveVector {
    match_each_integer_ptype!(array.ptype(), |T| {
        unpack_into_pvector::<T>(array, selection_mask).into()
    })
}

/// Given a [`BitPackedArray`] and a template type `T: BitPacked`, unpacks all bitpacked values and
/// creates a new [`PVector<T>`].
///
/// If a selection mask is passed in, the resultant vector will have the specified values filtered
/// out.
///
/// Internally, this calls [`write_unpacked_to_pvector`].
pub fn unpack_into_pvector<T: BitPacked>(
    array: &BitPackedArray,
    selection_mask: Option<&Mask>,
) -> PVector<T> {
    let num_new_values = true_count_of_optional_mask(&selection_mask, array.len());
    let mut pvector = PVectorMut::with_capacity(num_new_values);

    if num_new_values == 0 {
        return pvector.freeze();
    }

    write_unpacked_to_pvector(array, selection_mask, &mut pvector);

    pvector.freeze()
}

/// Given a [`BitPackedArray`] and a template type `T: BitPacked`, unpacks all bitpacked values and
/// writes them directly into an existing [`PVectorMut<T>`].
///
/// If a selection mask is passed in, the specified values will not be written.
pub fn write_unpacked_to_pvector<T: BitPacked>(
    array: &BitPackedArray,
    selection_mask: Option<&Mask>,
    vector: &mut PVectorMut<T>,
) {
    let num_new_values = true_count_of_optional_mask(&selection_mask, array.len());
    if num_new_values == 0 {
        return;
    }

    let validity_mask = match selection_mask {
        Some(selection_mask) => array.validity_mask().filter(selection_mask),
        None => array.validity_mask(),
    };
    debug_assert_eq!(validity_mask.len(), num_new_values);

    // SAFETY: We add the same amount of elements to both the buffer and the validity mask.
    let (buffer_mut, vector_validity_mut) = unsafe { vector.mut_parts() };

    // We need to write the unpacked values to the buffer as well as update the validity mask.
    write_unpacked_to_buffer(array, selection_mask, buffer_mut);
    vector_validity_mut.append_mask(&validity_mask);

    debug_assert_eq!(buffer_mut.len(), vector_validity_mut.len());
}

/// Given a [`BitPackedArray`] and a template type `T: BitPacked`, unpacks all bitpacked values and
/// writes them directly into an existing [`BufferMut<T>`].
///
/// If a selection mask is passed in, the specified values will not be written.
///
/// WARNING: This function will completely ignore the validity mask of the [`BitPackedArray`], so
/// this should only be called from [`write_unpacked_to_pvector`].
pub fn write_unpacked_to_buffer<T: BitPacked>(
    array: &BitPackedArray,
    selection_mask: Option<&Mask>,
    buffer: &mut BufferMut<T>,
) {
    let num_new_values = true_count_of_optional_mask(&selection_mask, array.len());
    if num_new_values == 0 {
        return;
    }

    let old_buffer_len = buffer.len();
    buffer.reserve(num_new_values);

    // We will be unpacking values directly into the uninitialized region of the buffer.
    let buffer_uninit_slice = &mut buffer.spare_capacity_mut()[..num_new_values];

    // If the selection mask is sparse, then we want to filter the bitpacked values while we are
    // unpacking all of the values.
    if let Some(selection) = selection_mask
        && selection.density() < unpack_then_filter_threshold::<T>()
    {
        filter_while_unpacking_array(array, selection, buffer_uninit_slice);

        // SAFETY: `filter_while_unpacking_array` writes exactly `num_new_values` values into the
        // buffer, so we know that all values up to the new length are initialized.
        unsafe { buffer.set_len(old_buffer_len + num_new_values) };

        return;
    }

    // Otherwise, if the selection mask is dense, then we might as well unpack all of the values and
    // then perform filtering.
    unpack_array(array, buffer_uninit_slice);

    // SAFETY: `unpack_array` fully unpacks the bitpacked array and writes `array.len()` values into
    // the buffer, so we know that all values up to the new length are initialized.
    unsafe { buffer.set_len(old_buffer_len + array.len()) };

    // Now that the array has been unpacked, apply the filter in-place.
    if let Some(selection) = selection_mask {
        buffer.filter(selection)
    }
}

/// Unpacks the bitpacked values in the [`BitPackedArray`] directly into a mutable buffer.
///
/// On return, all values in the given buffer will have been initialized.
///
/// Note that the caller should probably ensure that there array isn't empty and that the true count
/// of the selection mask isn't 0 for performance purposes.
///
/// WARNING: This function will completely ignore the validity mask of the [`BitPackedArray`], so
/// this should only be called from [`write_unpacked_to_pvector`].
fn unpack_array<T: BitPacked>(array: &BitPackedArray, buffer: &mut [MaybeUninit<T>]) {
    todo!()
}

/// Unpacks the bitpacked array into the given buffer according to the given selection mask.
///
/// On return, all values in the given buffer will have been initialized.
///
/// Note that the caller should probably ensure that there array isn't empty and that the true count
/// of the selection mask isn't 0 for performance purposes.
///
/// WARNING: This function will completely ignore the validity mask of the [`BitPackedArray`], so
/// this should only be called from [`write_unpacked_to_pvector`].
fn filter_while_unpacking_array<T: BitPacked>(
    array: &BitPackedArray,
    selection_mask: &Mask,
    buffer: &mut [MaybeUninit<T>],
) {
    todo!()
}
