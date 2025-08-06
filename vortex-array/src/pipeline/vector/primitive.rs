// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bits::BitVector;
use crate::pipeline::types::Element;
use crate::pipeline::view::ViewMut;
use vortex_dtype::NativePType;

pub struct PrimitiveVector<T> {
    elements: Vec<T>,
    validity: Option<BitVector>, // TODO(ngates): is this optional? Or just always allocate?
}

impl<T: NativePType> Default for PrimitiveVector<T> {
    fn default() -> Self {
        PrimitiveVector {
            elements: vec![T::default(); crate::pipeline::N],
            validity: None,
        }
    }
}

impl<T: Element> PrimitiveVector<T> {
    pub fn as_view_mut(&mut self) -> ViewMut<'_> {
        ViewMut::new::<T>(
            &mut self.elements,
            self.validity.as_mut().map(|v| v.as_view_mut()),
        )
    }
}

impl<T: NativePType> AsRef<[T]> for PrimitiveVector<T> {
    fn as_ref(&self) -> &[T] {
        &self.elements
    }
}

impl<T: NativePType> AsMut<[T]> for PrimitiveVector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }
}
