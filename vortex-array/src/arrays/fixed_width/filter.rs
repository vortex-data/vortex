// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::MaskValues;
use vortex_mask::MaskValuesRef;

use super::FixedWidthArray;
use super::with_values;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::filter::filter_buffer;
use crate::arrays::filter::filter_validity;
use crate::dtype::i256;

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests;

pub(crate) fn filter<V: FixedWidthArray>(
    array: &Array<V>,
    mask: &MaskValuesRef,
    allocator: &BufferAllocatorRef,
) -> Array<V> {
    let array = array.as_view();
    let values = match V::byte_width(array) {
        1 => filter_records::<V, u8>(array, mask.as_ref(), allocator),
        2 => filter_records::<V, u16>(array, mask.as_ref(), allocator),
        4 => filter_records::<V, u32>(array, mask.as_ref(), allocator),
        8 => filter_records::<V, u64>(array, mask.as_ref(), allocator),
        16 => filter_records::<V, u128>(array, mask.as_ref(), allocator),
        32 => filter_records::<V, i256>(array, mask.as_ref(), allocator),
        byte_width => vortex_panic!("Unsupported fixed-width byte width: {byte_width}"),
    };
    let validity = filter_validity(
        array
            .validity()
            .vortex_expect("validity is derivable for a valid fixed-width array"),
        mask,
    );
    with_values(array, values, mask.true_count(), validity)
        .vortex_expect("filtering fixed-width values preserves array invariants")
}

fn filter_records<V: FixedWidthArray, T: Copy>(
    array: ArrayView<'_, V>,
    mask: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> ByteBuffer {
    let values = V::values::<T>(array);
    let alignment = values.alignment();
    // `filter_buffer` picks between in-place compaction, cached indices/slices,
    // byte-compress, and bitmap iteration based on record width and mask density.
    filter_buffer(values, mask, allocator)
        .into_byte_buffer()
        .aligned(alignment)
}
