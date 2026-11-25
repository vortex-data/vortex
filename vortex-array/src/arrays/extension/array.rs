// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Extension;
use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;

#[array_slots(Extension)]
pub struct ExtensionSlots {
    /// The backing storage array for this extension array.
    #[slot(0)]
    pub storage: ArrayRef,
}

pub trait ExtensionArrayExt: TypedArrayRef<Extension> {
    fn ext_dtype(&self) -> &ExtDTypeRef {
        self.dtype()
            .as_extension_opt()
            .vortex_expect("extension array somehow did not have an extension dtype")
    }

    /// Returns the backing storage array.
    fn storage_array(&self) -> &ArrayRef {
        self.storage()
    }
}
impl<T: TypedArrayRef<Extension>> ExtensionArrayExt for T {}

impl Array<Extension> {
    /// Reconstructs an extension array from stored metadata and its storage array.
    ///
    /// See [`try_from_storage_with_vtable`](Self::try_from_storage_with_vtable) for the checks.
    pub fn try_from_storage<V: ExtVTable + Default>(
        stored_metadata: V::StoredMetadata,
        storage: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Self::try_from_storage_with_vtable(V::default(), stored_metadata, storage, ctx)
    }

    /// Reconstructs metadata and checks the storage array against the extension type.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata reconstruction or validation fails, or if
    /// [`try_new`](Self::try_new) rejects the storage.
    pub fn try_from_storage_with_vtable<V: ExtVTable>(
        vtable: V,
        stored_metadata: V::StoredMetadata,
        storage: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let dtype = ExtDType::try_from_storage_with_vtable(
            vtable,
            stored_metadata,
            storage.dtype().clone(),
        )?;

        Self::try_new(dtype.erased(), storage, ctx)
    }

    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if [`try_new`](Self::try_new) rejects the storage.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef, ctx: &mut ExecutionCtx) -> Self {
        Self::try_new(ext_dtype, storage_array, ctx)
            .vortex_expect("Unable to create `ExtensionArray`")
    }

    /// Constructs an extension array after checking its storage dtype and non-null values.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtype differs, including top-level or nested nullability,
    /// if reading the storage fails, or if any non-null value is outside the extension's domain.
    pub fn try_new(
        ext_dtype: ExtDTypeRef,
        storage_array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let array = Self::try_new_structural(ext_dtype, storage_array)?;
        array
            .ext_dtype()
            .validate_storage_array(array.storage_array(), ctx)?;

        Ok(array)
    }

    /// Wraps storage after checking that its dtype matches the extension's storage declaration.
    ///
    /// The values are as legal as the source data. Callers that require legal values must check
    /// the whole array with [`super::validate_extension_values`]. Imports use this constructor
    /// because a child cannot tell which of its rows a null ancestor masks.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtypes differ, including top-level or nested nullability.
    pub fn try_new_structural(ext_dtype: ExtDTypeRef, storage: ArrayRef) -> VortexResult<Self> {
        Self::try_from_parts(Self::storage_parts(ext_dtype, storage))
    }

    /// Creates a new `ExtensionArray` without validating its storage array.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `storage_array` has the extension dtype's storage dtype.
    ///
    /// Each visible non-null value must be legal for the extension type or come unchanged from
    /// an extension array with the same dtype, apart from top-level nullability. A null ancestor
    /// can mask a placeholder storage value. Breaking this value requirement can cause incorrect
    /// results or read errors but does not cause undefined behavior.
    pub unsafe fn new_unchecked(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        // SAFETY: the caller guarantees the storage dtype, and the length comes from the child.
        unsafe { Self::from_parts_unchecked(Self::storage_parts(ext_dtype, storage_array)) }
    }

    fn storage_parts(ext_dtype: ExtDTypeRef, storage: ArrayRef) -> ArrayParts<Extension> {
        ArrayParts::new(
            Extension,
            DType::Extension(ext_dtype),
            storage.len(),
            EmptyArrayData,
            ExtensionSlots { storage }.into_slots(),
        )
    }

    /// Creates a new [`ExtensionArray`](crate::arrays::ExtensionArray) from a vtable, metadata, and
    /// a storage array.
    ///
    /// Uses the storage array's top-level nullability when constructing the dtype. The complete
    /// storage dtype must match the metadata's declaration.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata validation fails or [`try_new`](Self::try_new) rejects the
    /// storage.
    pub fn try_new_from_vtable<V: ExtVTable>(
        vtable: V,
        metadata: V::Metadata,
        storage_array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let ext_dtype =
            ExtDType::<V>::try_with_vtable(vtable, metadata, storage_array.dtype().nullability())?
                .erased();

        Self::try_new(ext_dtype, storage_array, ctx)
    }
}
