// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector extension type for fixed-length float vectors (e.g., embeddings).

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
use vortex_error::vortex_bail;

/// Reads the [`Vector`] metadata from `storage`, a `FixedSizeList` of floats.
///
/// This does not check the element nullability. The [`Vector`] storage dtype declares
/// non-nullable elements, so the comparison with the declaration rejects nullable ones.
pub(crate) fn vector_metadata(storage: &DType) -> VortexResult<VectorMatcherMetadata> {
    let DType::FixedSizeList(element_dtype, dimensions, _nullability) = storage else {
        vortex_bail!("Vector storage dtype must be a FixedSizeList, got {storage}");
    };
    let DType::Primitive(element_ptype, _) = element_dtype.as_ref() else {
        vortex_bail!("Vector element dtype must be a float, got {element_dtype}");
    };

    VectorMatcherMetadata::try_new(*element_ptype, *dimensions)
}

/// The Vector extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector;

impl Vector {
    /// Helper function for creating a new [`Vector`] [`ExtensionArray`].
    ///
    /// # Errors
    ///
    /// Returns an error if the storage array does not have the [`Vector`] storage dtype.
    pub fn try_new_vector_array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let metadata = vector_metadata(storage.dtype())?;
        ExtensionArray::try_new_from_vtable(Vector, metadata, storage).map(|ext| ext.into_array())
    }

    /// Helper function to build a [`Vector`] [`ExtensionArray`] whose storage is a
    /// [`ConstantArray`], broadcasting a single vector `elements` across `len` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Vector`] extension dtype rejects the constructed storage dtype.
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

mod arrow;
mod matcher;

pub use arrow::ARROW_VECTOR_EXTENSION_NAME;
pub use matcher::AnyVector;
pub use matcher::VectorMatcherMetadata;

mod vtable;
