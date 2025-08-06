// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::N;
use crate::pipeline::bits::BitView;
use crate::pipeline::types::Element;

pub struct VectorMut<'a, E: Element> {
    elements: &'a mut [E; N],
}

impl<'a, E: Element> VectorMut<'a, E> {
    pub fn select_mask(&mut self, _mask: BitView) {
        todo!()
    }
}

impl<E: Element> AsRef<[E; N]> for VectorMut<'_, E> {
    fn as_ref(&self) -> &[E; N] {
        self.elements
    }
}

impl<E: Element> AsMut<[E; N]> for VectorMut<'_, E> {
    fn as_mut(&mut self) -> &mut [E; N] {
        self.elements
    }
}
