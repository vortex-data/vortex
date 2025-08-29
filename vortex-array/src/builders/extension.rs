// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::ExtScalar;

use crate::arrays::ExtensionArray;
use crate::builders::{
    ArrayBuilder, ArrayBuilderExt, DEFAULT_BUILDER_CAPACITY, builder_can_be_extended_by,
    builder_with_capacity,
};
use crate::{Array, ArrayRef, IntoArray};

pub struct ExtensionBuilder {
    dtype: DType,
    storage: Box<dyn ArrayBuilder>,
}

impl ExtensionBuilder {
    /// Creates a new `ExtensionBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(ext_dtype: Arc<ExtDType>) -> Self {
        Self::with_capacity(ext_dtype, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `DecimalBuilder` with the given `capacity`.
    pub fn with_capacity(ext_dtype: Arc<ExtDType>, capacity: usize) -> Self {
        Self {
            storage: builder_with_capacity(ext_dtype.storage_dtype(), capacity),
            dtype: DType::Extension(ext_dtype),
        }
    }

    /// Appends a `value` with an extension type to the builder.
    pub fn append_ext(&mut self, value: ExtScalar) -> VortexResult<()> {
        self.storage.append_scalar(&value.storage())
    }

    /// Appends a optional `value` (representing a nullable value) with an extension type to the
    /// builder.
    ///
    /// # Panics
    ///
    /// This method will panic if the input is `None` and the builder is non-nullable.
    pub fn append_ext_opt(&mut self, value: Option<ExtScalar>) -> VortexResult<()> {
        match value {
            Some(value) => self.append_ext(value),
            None => {
                self.append_null();
                Ok(())
            }
        }
    }

    /// Finishes the builder directly into an [`ExtensionArray`].
    pub fn finish_into_extension(&mut self) -> ExtensionArray {
        let storage = self.storage.finish();
        ExtensionArray::new(self.ext_dtype().clone(), storage)
    }

    /// The [`ExtDType`] of this builder.
    fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext_dtype) = &self.dtype else {
            vortex_panic!("`ExtensionBuilder` somehow had dtype {}", self.dtype)
        };

        ext_dtype
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

    fn append_nulls(&mut self, n: usize) {
        self.storage.append_nulls(n)
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        assert!(
            builder_can_be_extended_by(&self.dtype, array.dtype()),
            "tried to extend a builder with an array of different `DType`"
        );

        let array = array.to_canonical()?.into_extension()?;

        array.storage().append_to_builder(self.storage.as_mut())
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
