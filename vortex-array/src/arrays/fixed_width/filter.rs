// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_mask::MaskValues;
use vortex_mask::MaskValuesRef;

use super::FixedWidthArray;
use super::match_each_record_width;
use super::with_values;
use crate::array::Array;
use crate::arrays::filter::filter_buffer;
use crate::arrays::filter::filter_buffer_byte_compress;
use crate::arrays::filter::filter_validity;

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests;

pub(crate) fn filter<V: FixedWidthArray>(array: &Array<V>, mask: &MaskValuesRef) -> Array<V> {
    let array = array.as_view();
    let values = filter_records(V::values(array), V::byte_width(array), mask.as_ref());
    let validity = filter_validity(
        array
            .validity()
            .vortex_expect("validity is derivable for a valid fixed-width array"),
        mask,
    );
    with_values(array, values, mask.true_count(), validity)
        .vortex_expect("filtering fixed-width values preserves array invariants")
}

fn filter_records(values: ByteBuffer, byte_width: usize, mask: &MaskValues) -> ByteBuffer {
    let alignment = values.alignment();

    match_each_record_width!(
        byte_width,
        |W| {
            let records = Buffer::<[u8; W]>::from_byte_buffer(values);
            // Byte-compress processes eight records per mask byte and wins for narrow records,
            // while wider records move enough bytes each for the plain filter to win.
            let filtered = if W <= 4 {
                filter_buffer_byte_compress(records, mask)
            } else {
                filter_buffer(records, mask)
            };
            filtered.into_byte_buffer().aligned(alignment)
        },
        _ => {
            match values.try_into_mut() {
                Ok(mut values) => {
                    let mut destination = 0;
                    mask.bit_buffer().for_each_set_index(|index| {
                        let source = index * byte_width;
                        values.copy_within(source..source + byte_width, destination);
                        destination += byte_width;
                    });
                    values.truncate(destination);
                    values.freeze().into_byte_buffer().aligned(alignment)
                }
                Err(values) => {
                    let mut filtered = BufferMut::with_capacity(mask.true_count() * byte_width);
                    mask.bit_buffer().for_each_set_index(|index| {
                        let start = index * byte_width;
                        filtered.extend_from_slice(&values[start..start + byte_width]);
                    });
                    filtered.freeze().into_byte_buffer().aligned(alignment)
                }
            }
        }
    )
}
