// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_buffer::filter;
use vortex_error::VortexExpect;
use vortex_mask::MaskValues;
use vortex_mask::MaskValuesRef;

use super::FixedWidthArray;
use super::with_values;
use crate::array::Array;
use crate::arrays::filter::filter_validity;

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests;

pub(crate) fn filter<V: FixedWidthArray>(
    array: &Array<V>,
    mask: &MaskValuesRef,
    allocator: &BufferAllocatorRef,
) -> Array<V> {
    let array = array.as_view();
    let values = filter_records(
        V::values(array),
        V::byte_width(array),
        mask.as_ref(),
        allocator,
    );
    let validity = filter_validity(
        array
            .validity()
            .vortex_expect("validity is derivable for a valid fixed-width array"),
        mask,
    );
    with_values(array, values, mask.true_count(), validity)
        .vortex_expect("filtering fixed-width values preserves array invariants")
}

fn filter_records(
    values: ByteBuffer,
    byte_width: usize,
    mask: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> ByteBuffer {
    filter::filter_records(values, byte_width, mask.selection(), allocator)
}
