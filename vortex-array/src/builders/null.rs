// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_dtype::DType;
use vortex_mask::Mask;

use crate::arrays::NullArray;
use crate::builders::ArrayBuilder;
use crate::{Array, ArrayRef, IntoArray};

/// The builder for building a [`NullArray`].
pub struct NullBuilder {
    length: usize,
}

impl Default for NullBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NullBuilder {
    pub fn new() -> Self {
        Self { length: 0 }
    }
}

impl ArrayBuilder for NullBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &DType::Null
    }

    fn len(&self) -> usize {
        self.length
    }

    fn append_zeros(&mut self, n: usize) {
        self.length += n;
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.length += n;
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        self.append_nulls(array.len());
    }

    fn ensure_capacity(&mut self, _capacity: usize) {}

    fn set_validity(&mut self, validity: Mask) {
        self.length = validity.len();
    }

    fn finish(&mut self) -> ArrayRef {
        NullArray::new(self.length).into_array()
    }
}
