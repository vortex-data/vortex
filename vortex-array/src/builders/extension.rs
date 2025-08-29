// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::ExtScalar;

use crate::arrays::ExtensionArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, builder_with_capacity};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// The builder for building a [`ExtensionArray`].
pub struct ExtensionBuilder {
    dtype: DType,
    storage: Box<dyn ArrayBuilder>,
}

impl ExtensionBuilder {
    /// Creates a new `ExtensionBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(ext_dtype: Arc<ExtDType>) -> Self {
        Self::with_capacity(ext_dtype, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `ExtensionBuilder` with the given `capacity`.
    pub fn with_capacity(ext_dtype: Arc<ExtDType>, capacity: usize) -> Self {
        Self {
            storage: builder_with_capacity(ext_dtype.storage_dtype(), capacity),
            dtype: DType::Extension(ext_dtype),
        }
    }

    /// Appends an extension `value` to the builder.
    pub fn append_value(&mut self, value: ExtScalar) -> VortexResult<()> {
        self.storage.append_scalar(&value.storage())
    }

    /// Appends an optional extension (representing a nullable extension value) to the builder.
    ///
    /// # Panics
    ///
    /// This method will panic if the input is `None` and the builder is non-nullable.
    pub fn append_option(&mut self, value: Option<ExtScalar>) -> VortexResult<()> {
        match value {
            Some(value) => self.append_value(value),
            None => {
                self.append_nulls(1);
                Ok(())
            }
        }
    }

    /// Finishes the builder directly into a [`ExtensionArray`].
    pub fn finish_into_extension(&mut self) -> ExtensionArray {
        let storage = self.storage.finish();
        ExtensionArray::new(self.ext_dtype(), storage)
    }

    /// The [`ExtDType`] of this builder.
    fn ext_dtype(&self) -> Arc<ExtDType> {
        if let DType::Extension(ext_dtype) = &self.dtype {
            ext_dtype.clone()
        } else {
            unreachable!()
        }
    }
}

impl ArrayBuilder for ExtensionBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.storage.append_zeros(n)
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.storage.append_nulls(n)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        self.storage
            .extend_from_array(array.to_extension().storage())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.storage.ensure_capacity(capacity)
    }

    fn set_validity(&mut self, validity: Mask) {
        self.storage.set_validity(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_extension().into_array()
    }
}
