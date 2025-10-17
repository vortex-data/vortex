// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::{VortexResult, vortex_ensure};
use vortex_mask::Mask;
use vortex_scalar::{ExtScalar, Scalar};

use crate::arrays::ExtensionArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, builder_with_capacity};
use crate::canonical::{Canonical, ToCanonical};
use crate::{Array, ArrayRef, IntoArray};

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

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "ExtensionBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        let ext_scalar = ExtScalar::try_from(scalar)?;
        self.append_value(ext_scalar)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let ext_array = array.to_extension();
        self.storage.extend_from_array(ext_array.storage())
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.storage.reserve_exact(capacity)
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        unsafe { self.storage.set_validity_unchecked(validity) };
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_extension().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Extension(self.finish_into_extension())
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{ExtDType, ExtID, Nullability};
    use vortex_scalar::Scalar;

    use super::*;
    use crate::builders::ArrayBuilder;

    #[test]
    fn test_append_scalar() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(
                vortex_dtype::PType::I32,
                Nullability::Nullable,
            )),
            None,
        ));

        let mut builder = ExtensionBuilder::new(ext_dtype.clone());

        // Test appending a valid extension value.
        let storage1 = Scalar::from(42i32);
        let ext_scalar1 = Scalar::extension(ext_dtype.clone(), storage1);
        builder.append_scalar(&ext_scalar1).unwrap();

        // Test appending another value.
        let storage2 = Scalar::from(84i32);
        let ext_scalar2 = Scalar::extension(ext_dtype.clone(), storage2);
        builder.append_scalar(&ext_scalar2).unwrap();

        // Test appending null value.
        let null_storage = Scalar::null(DType::Primitive(
            vortex_dtype::PType::I32,
            Nullability::Nullable,
        ));
        let null_scalar = Scalar::extension(ext_dtype.clone(), null_storage);
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_extension();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.

        let scalar0 = array.scalar_at(0);
        let ext0 = scalar0.as_extension();
        assert_eq!(ext0.storage().as_primitive().typed_value::<i32>(), Some(42));

        let scalar1 = array.scalar_at(1);
        let ext1 = scalar1.as_extension();
        assert_eq!(ext1.storage().as_primitive().typed_value::<i32>(), Some(84));

        let scalar2 = array.scalar_at(2);
        let ext2 = scalar2.as_extension();
        assert_eq!(ext2.storage().as_primitive().typed_value::<i32>(), None); // Storage is null.

        // Test wrong dtype error.
        let mut builder = ExtensionBuilder::new(ext_dtype);
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
