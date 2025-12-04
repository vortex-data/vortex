// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::NullBuffer;
use vortex_mask::Mask;

impl From<Mask> for Option<NullBuffer> {
    fn from(value: Mask) -> Self {
        match value {
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
        }
    }
}
