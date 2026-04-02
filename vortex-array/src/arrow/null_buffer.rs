// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use arrow_buffer::NullBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::validity::Validity;

/// Converts a [`Validity`] to an Arrow [`NullBuffer`], executing the validity array if needed.
pub fn to_arrow_null_buffer(
    validity: Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<NullBuffer>> {
    Ok(match validity {
        Validity::NonNullable | Validity::AllValid => None,
        Validity::AllInvalid => Some(NullBuffer::new_null(len)),
        Validity::Array(array) => to_null_buffer(array.execute::<Mask>(ctx)?),
    })
}

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

    use crate::arrow::null_buffer::to_null_buffer;

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
