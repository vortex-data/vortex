// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::{BooleanBuffer, NullBuffer};
use vortex_mask::Mask;

/// Converts a mask to a null buffer.
pub fn to_null_buffer(mask: Mask) -> Option<NullBuffer> {
    match mask {
        Mask::AllTrue(_) => None,
        Mask::AllFalse(l) => Some(NullBuffer::new_null(l)),
        Mask::Values(values) => Some(NullBuffer::from(BooleanBuffer::from(
            values.bit_buffer().clone(),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;

    use crate::arrow::compute::to_arrow::null_buffer::to_null_buffer;

    #[test]
    fn test_mask_to_null_buffer() {
        let all_true = Mask::new_true(5);
        assert!(to_null_buffer(all_true).is_none());

        let all_false = Mask::new_false(5);
        let null_buffer = to_null_buffer(all_false).unwrap();
        assert_eq!(null_buffer.null_count(), 5);

        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let null_buffer = to_null_buffer(values).unwrap();
        assert_eq!(null_buffer.null_count(), 2);
    }
}
