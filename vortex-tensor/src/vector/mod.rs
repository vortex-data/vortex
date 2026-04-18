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
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

/// The Vector extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector;

impl Vector {
    /// Wrap a `FixedSizeList`-valued `storage` array in a [`Vector`] extension array.
    ///
    /// The storage's dtype is reused verbatim for the extension's storage dtype, so the caller
    /// is responsible for having already constructed an FSL with the float element ptype and
    /// non-nullable elements that [`Vector::validate_dtype`](ExtVTable::validate_dtype) requires.
    ///
    /// [`ExtVTable::validate_dtype`]: vortex_array::dtype::extension::ExtVTable::validate_dtype
    ///
    /// # Errors
    ///
    /// Returns an error if `storage` does not satisfy [`Vector`]'s storage-dtype contract (e.g.
    /// it is not a `FixedSizeList` of non-nullable floats).
    pub fn wrap_storage(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let ext_dtype = ExtDType::<Self>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    /// Build a [`Vector`] extension array whose storage is a [`ConstantArray`] broadcasting a
    /// single vector `elements` across `len` rows.
    ///
    /// This is the array shape that [`CosineSimilarity::try_new_array`] and similar binary tensor
    /// scalar functions expect for the constant-query side of a database-vs-query scan: the inner
    /// `ScalarFnArray` contract requires both children to have the same length, so the query is
    /// broadcast rather than represented as a literal length-1 input.
    ///
    /// [`CosineSimilarity::try_new_array`]: crate::scalar_fns::cosine_similarity::CosineSimilarity::try_new_array
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Vector`] extension dtype rejects the constructed storage dtype.
    pub fn constant_array<T: NativePType + Into<PValue>>(
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
        Self::wrap_storage(ConstantArray::new(storage_scalar, len).into_array())
    }
}

#[cfg(test)]
mod ctor_tests {
    use vortex_array::arrays::Extension;

    use super::*;

    #[test]
    fn constant_array_produces_vector_extension() {
        let array = Vector::constant_array(&[1.0f32, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(array.len(), 5);
        assert!(array.as_opt::<Extension>().is_some());
    }
}

mod matcher;

pub use matcher::AnyVector;
pub use matcher::VectorMatcherMetadata;

mod vtable;
