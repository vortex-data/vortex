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
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Validates that `storage` is a valid storage dtype for a [`Vector`] or
/// [`NormalizedVector`](crate::normalized_vector::NormalizedVector) extension type.
///
/// The storage must be a `FixedSizeList<float, dim, nullability>` with non-nullable float
/// elements. The outer nullability is not constrained.
pub(crate) fn validate_vector_storage_dtype(storage: &DType) -> VortexResult<()> {
    let DType::FixedSizeList(element_dtype, _list_size, _nullability) = storage else {
        vortex_bail!("Vector storage dtype must be a FixedSizeList, got {storage}");
    };

    vortex_ensure!(
        element_dtype.is_float(),
        "Vector element dtype must be a float, got {element_dtype}"
    );
    vortex_ensure!(
        !element_dtype.is_nullable(),
        "Vector element dtype must be non-nullable"
    );

    Ok(())
}

/// The Vector extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector;

impl Vector {
    /// Helper function for creating a new [`Vector`] [`ExtensionArray`].
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Vector`] extension dtype rejects the storage array.
    pub fn try_new_vector_array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, storage)
            .map(|ext| ext.into_array())
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

mod matcher;

pub use matcher::AnyVector;
pub use matcher::VectorMatcherMetadata;

mod vtable;
