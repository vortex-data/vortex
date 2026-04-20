// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::builder_with_capacity;
use crate::canonical::Canonical;
#[expect(deprecated)]
use crate::canonical::ToCanonical as _;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::ExtScalar;
use crate::scalar::Scalar;

/// The builder for building a [`ExtensionArray`].
pub struct ExtensionBuilder {
    dtype: DType,
    storage: Box<dyn ArrayBuilder>,
}

impl ExtensionBuilder {
    /// Creates a new `ExtensionBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(ext_dtype: ExtDTypeRef) -> Self {
        Self::with_capacity(ext_dtype, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `ExtensionBuilder` with the given `capacity`.
    pub fn with_capacity(ext_dtype: ExtDTypeRef, capacity: usize) -> Self {
        Self {
            storage: builder_with_capacity(ext_dtype.storage_dtype(), capacity),
            dtype: DType::Extension(ext_dtype),
        }
    }

    /// Appends an extension `value` to the builder.
    pub fn append_value(&mut self, value: ExtScalar) -> VortexResult<()> {
        self.storage.append_scalar(&value.to_storage_scalar())
    }

    /// Finishes the builder directly into a [`ExtensionArray`].
    pub fn finish_into_extension(&mut self) -> ExtensionArray {
        let storage = self.storage.finish();
        ExtensionArray::new(self.ext_dtype(), storage)
    }

    /// The [`ExtDType`] of this builder.
    fn ext_dtype(&self) -> ExtDTypeRef {
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
            "ExtensionBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        self.append_value(scalar.as_extension())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        #[expect(deprecated)]
        let ext_array = array.to_extension();
        self.storage.extend_from_array(ext_array.storage_array())
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
    use super::*;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::Scalar;

    #[test]
    fn test_append_scalar() {
        let ext_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();

        let mut builder = ExtensionBuilder::new(ext_dtype.clone());

        // Test appending a valid extension value.
        let storage1 = Scalar::from(Some(42i32));
        let ext_scalar1 = Scalar::extension::<Date>(TimeUnit::Days, storage1);
        builder.append_scalar(&ext_scalar1).unwrap();

        // Test appending another value.
        let storage2 = Scalar::from(Some(84i32));
        let ext_scalar2 = Scalar::extension::<Date>(TimeUnit::Days, storage2);
        builder.append_scalar(&ext_scalar2).unwrap();

        // Test appending null value.
        let null_storage = Scalar::null(DType::Primitive(
            crate::dtype::PType::I32,
            Nullability::Nullable,
        ));
        let null_scalar = Scalar::extension::<Date>(TimeUnit::Days, null_storage);
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_extension();
        let expected = ExtensionArray::new(
            ext_dtype.clone(),
            PrimitiveArray::from_option_iter([Some(42i32), Some(84), None]).into_array(),
        );

        assert_arrays_eq!(&array, &expected);
        assert_eq!(array.len(), 3);

        // Test wrong dtype error.
        let mut builder = ExtensionBuilder::new(ext_dtype);
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
