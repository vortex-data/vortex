// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-length floating-point vectors, such as embeddings.
//!
//! [`VectorMetadata`] declares the element type and dimension. Storage is a fixed-size list of
//! non-nullable floats, with an independently nullable outer list.

mod arrow;
pub use arrow::ARROW_VECTOR_EXTENSION_NAME;

mod matcher;
pub use matcher::AnyVector;

mod metadata;
pub use metadata::VectorMetadata;

mod vtable;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

/// The Vector extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector;

impl Vector {
    /// Wraps a fixed-size list of non-nullable floats in a [`Vector`] extension array.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage array does not have the [`Vector`] storage dtype.
    pub fn try_new_vector_array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        ExtensionArray::try_from_storage::<Vector>((), storage).map(|ext| ext.into_array())
    }

    /// Broadcasts `elements` as a vector across `len` rows using [`ConstantArray`] storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the element type is not a float.
    pub(crate) fn constant_array<T: NativePType + Into<PValue>>(
        elements: &[T],
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children: Vec<Scalar> = elements
            .iter()
            .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
            .collect();
        let storage_scalar =
            Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
        Self::try_new_vector_array(ConstantArray::new(storage_scalar, len).into_array())
    }
}
