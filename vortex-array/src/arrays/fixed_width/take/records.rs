// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::take_values;
use crate::arrays::fixed_width::match_each_record_width;
use crate::dtype::UnsignedPType;

pub(super) fn take_byte_records<I: UnsignedPType>(
    values: &ByteBuffer,
    byte_width: usize,
    record_count: usize,
    indices: &[I],
) -> VortexResult<ByteBuffer> {
    let alignment = values.alignment();

    match_each_record_width!(
        byte_width,
        |W| {
            let records = Buffer::<[u8; W]>::from_byte_buffer(values.clone());
            debug_assert_eq!(records.len(), record_count);
            Ok(take_values(records.as_slice(), indices)
                .into_byte_buffer()
                .aligned(alignment))
        },
        _ => {
            let output_len = indices
                .len()
                .checked_mul(byte_width)
                .ok_or_else(|| vortex_err!("Fixed-width take output length overflows usize"))?;
            let mut result = BufferMut::<u8>::with_capacity(output_len);
            for index in indices {
                let index = index.as_();
                assert!(
                    index < record_count,
                    "take index {index} out of bounds for length {record_count}"
                );
                let start = index * byte_width;
                result.extend_from_slice(&values[start..start + byte_width]);
            }
            Ok(result.freeze().into_byte_buffer().aligned(alignment))
        }
    )
}
