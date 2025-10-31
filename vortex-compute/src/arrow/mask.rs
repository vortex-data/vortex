// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::NullBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrow::IntoArrow;

impl IntoArrow<Option<NullBuffer>> for Mask {
    fn into_arrow(self) -> VortexResult<Option<NullBuffer>> {
        Ok(match self {
            Mask::AllTrue(_) => None,
            Mask::AllFalse(len) => Some(NullBuffer::new_null(len)),
            Mask::Values(values) => {
                // SAFETY: we maintain our own validated true count.
                Some(unsafe {
                    NullBuffer::new_unchecked(
                        values.bit_buffer().clone().into(),
                        values.len() - values.true_count(),
                    )
                })
            }
        })
    }
}
