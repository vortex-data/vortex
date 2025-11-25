// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between `BitBuffer` and Arrow's `BooleanBuffer`.

use arrow_buffer::BooleanBuffer;

use crate::Alignment;
use crate::BitBuffer;
use crate::ByteBuffer;

impl From<BooleanBuffer> for BitBuffer {
    fn from(value: BooleanBuffer) -> Self {
        let offset = value.offset();
        let len = value.len();
        let buffer = value.into_inner();
        let buffer = ByteBuffer::from_arrow_buffer(buffer, Alignment::of::<u8>());

        BitBuffer::new_with_offset(buffer, len, offset)
    }
}

impl From<BitBuffer> for BooleanBuffer {
    fn from(value: BitBuffer) -> Self {
        let offset = value.offset();
        let len = value.len();
        let buffer = value.into_inner();

        BooleanBuffer::new(buffer.into_arrow_buffer(), offset, len)
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use arrow_buffer::BooleanBufferBuilder;

    use crate::BitBuffer;

    #[test]
    fn test_from_arrow() {
        let mut arrow_bools = BooleanBufferBuilder::new(10);
        arrow_bools.append_n(5, true);
        arrow_bools.append_n(5, false);
        let bit_buffer: BitBuffer = arrow_bools.finish().into();

        for i in 0..5 {
            assert!(bit_buffer.value(i));
        }

        for i in 5..10 {
            assert!(!bit_buffer.value(i));
        }

        // Convert back to Arrow
        let arrow_bools: BooleanBuffer = bit_buffer.into();

        for i in 0..5 {
            assert!(arrow_bools.value(i));
        }
        for i in 5..10 {
            assert!(!arrow_bools.value(i));
        }
    }
}
