// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::filter::Filter;
use vortex_buffer::BitView;

impl<'a, const NB: usize, T: Copy> Filter<BitView<'a, NB>> for &mut [T] {
    type Output = Self;

    fn filter(self, selection: &BitView<'a, NB>) -> Self::Output {
        todo!()
    }
}
