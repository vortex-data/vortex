// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::UnsignedPType;

use crate::take::Take;

impl<T: Copy, I: UnsignedPType> Take<[I]> for &Buffer<T> {
    type Output = Buffer<T>;

    fn take(self, indices: &[I]) -> Buffer<T> {
        self.as_slice().take(indices)
    }
}
