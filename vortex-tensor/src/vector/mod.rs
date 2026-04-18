// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector extension type for fixed-length float vectors (e.g., embeddings).

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
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
}

mod matcher;

pub use matcher::AnyVector;
pub use matcher::VectorMatcherMetadata;

mod vtable;
