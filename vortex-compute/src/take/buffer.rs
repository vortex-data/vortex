// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Unsigned;
use vortex_buffer::Buffer;

use crate::take::Take;

impl<T: Copy, I: Unsigned + AsPrimitive<usize>> Take<[I]> for &Buffer<T> {
    type Output = Buffer<T>;

    fn take(self, indices: &[I]) -> Buffer<T> {
        self.as_slice().take(indices)
    }
}
