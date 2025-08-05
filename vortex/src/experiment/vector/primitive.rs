// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::vector::BitVector;
use std::sync::Arc;

pub struct PrimitiveVector<T> {
    elements: Arc<[T; N]>,
    validity: Option<BitVector>, // TODO(ngates): is this optional? Or just always allocate?
}

impl<T> PrimitiveVector<T> {
    pub fn as_view(&self) -> PrimitiveView<'_, T> {
        PrimitiveView {
            elements: &self.elements,
            validity: self.validity.as_ref().map(|v| v.as_view()),
        }
    }
}
