// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBufferBuilder;
use num_traits::PrimInt;
use vortex_dtype::{NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::{ListViewArray, ListViewVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, PrimitiveBuilder};
use crate::compute::{TakeKernel, TakeKernelAdapter, take};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, OffsetPType, ToCanonical, register_kernel};

impl TakeKernel for ListViewVTable {
    #[allow(clippy::cognitive_complexity)]
    fn take(&self, array: &ListViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive();
        let offsets = array.offsets().to_primitive();
        let sizes = array.sizes().to_primitive();

        match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
            match_each_integer_ptype!(sizes.dtype().as_ptype(), |S| {
                match_each_integer_ptype!(indices.ptype(), |I| {
                    _take::<I, O, S>(
                        array,
                        offsets.as_slice::<O>(),
                        sizes.as_slice::<S>(),
                        &indices,
                        array.validity_mask(),
                        indices.validity_mask(),
                    )
                })
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(ListViewVTable).lift());

fn _take<I: NativePType, O: OffsetPType + NativePType + PrimInt, S: NativePType + PrimInt>(
    array: &ListViewArray,
    offsets: &[O],
    sizes: &[S],
    indices_array: &PrimitiveArray,
    data_validity: Mask,
    indices_validity_mask: Mask,
) -> VortexResult<ArrayRef> {
    let indices: &[I] = indices_array.as_slice::<I>();

    if !indices_validity_mask.all_true() || !data_validity.all_true() {
        return _take_nullable::<I, O, S>(
            array,
            offsets,
            sizes,
            indices,
            data_validity,
            indices_validity_mask,
        );
    }

    let mut new_offsets =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, indices.len());
    let mut new_sizes =
        PrimitiveBuilder::<S>::with_capacity(Nullability::NonNullable, indices.len());
    let mut elements_to_take =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = O::zero();

    for &data_idx in indices {
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        let start = offsets[data_idx];
        let size = sizes[data_idx];

        // Build the elements to take from the original elements array.
        let size_usize = size
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert size to usize: {}", size));

        elements_to_take.ensure_capacity(elements_to_take.len() + size_usize);
        for i in 0..size_usize {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < size"));
        }

        // Update the new offsets and sizes arrays.
        new_offsets.append_value(current_offset);
        new_sizes.append_value(size);
        current_offset = current_offset + O::from(size).vortex_expect("size fits in O");
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();
    let new_sizes = new_sizes.finish();

    let new_elements = take(array.elements(), elements_to_take.as_ref())?;

    // SAFETY: The arrays maintain the ListView invariants:
    // - Offsets and sizes have the same length as indices.
    // - Elements are properly taken based on offsets and sizes.
    // - Validity matches the original array's validity combined with indices validity.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            new_elements,
            new_offsets.to_array(),
            new_sizes.to_array(),
            indices_array
                .validity()
                .clone()
                .and(array.validity().clone()),
        )
    }
    .to_array())
}

fn _take_nullable<
    I: NativePType,
    O: OffsetPType + NativePType + PrimInt,
    S: NativePType + PrimInt,
>(
    array: &ListViewArray,
    offsets: &[O],
    sizes: &[S],
    indices: &[I],
    data_validity: Mask,
    indices_validity: Mask,
) -> VortexResult<ArrayRef> {
    let mut new_offsets =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, indices.len());
    let mut new_sizes =
        PrimitiveBuilder::<S>::with_capacity(Nullability::NonNullable, indices.len());
    let mut elements_to_take =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = O::zero();
    let mut new_validity = BooleanBufferBuilder::new(indices.len());

    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            // Index is null, so the output at this position is null.
            new_offsets.append_value(current_offset);
            new_sizes.append_zero();
            new_validity.append(false);
            continue;
        }

        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));

        if data_validity.value(data_idx) {
            // Both index and data are valid.
            let start = offsets[data_idx];
            let size = sizes[data_idx];

            let size_usize = size
                .to_usize()
                .unwrap_or_else(|| vortex_panic!("Failed to convert size to usize: {}", size));

            elements_to_take.ensure_capacity(elements_to_take.len() + size_usize);
            for i in 0..size_usize {
                elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < size"));
            }

            new_offsets.append_value(current_offset);
            new_sizes.append_value(size);
            current_offset = current_offset + O::from(size).vortex_expect("size fits in O");
            new_validity.append(true);
        } else {
            // Data at the index is null.
            new_offsets.append_value(current_offset);
            new_sizes.append_zero();
            new_validity.append(false);
        }
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();
    let new_sizes = new_sizes.finish();
    let new_elements = take(array.elements(), elements_to_take.as_ref())?;

    let new_validity: Validity = Validity::from(new_validity.finish());

    // SAFETY: The arrays maintain the ListView invariants:
    // - Offsets and sizes have the same length as indices.
    // - Elements are properly taken based on offsets and sizes.
    // - Validity correctly reflects null positions.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            new_elements,
            new_offsets.to_array(),
            new_sizes.to_array(),
            new_validity,
        )
    }
    .to_array())
}
