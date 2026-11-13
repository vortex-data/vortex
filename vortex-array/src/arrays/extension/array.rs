// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::EmptyArrayData;
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
    ) -> VortexResult<Self> {
        Self::try_from_storage_with_vtable(V::default(), stored_metadata, storage)
    }

    /// Reconstructs metadata and checks its declaration against the storage array's dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata reconstruction or validation fails, or storage differs from
    /// the declaration, including nested nullability.
    pub fn try_from_storage_with_vtable<V: ExtVTable>(
        vtable: V,
        stored_metadata: V::StoredMetadata,
        storage: ArrayRef,
    ) -> VortexResult<Self> {
        let dtype = ExtDType::try_from_storage_with_vtable(
            vtable,
            stored_metadata,
            storage.dtype().clone(),
        )?;

        Self::try_new(dtype.erased(), storage)
    }

    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array is not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        Self::try_new(ext_dtype, storage_array).vortex_expect("Unable to create `ExtensionArray`")
    }

    /// Constructs an extension array whose storage dtype exactly matches `ext_dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtypes differ, including top-level or nested nullability.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> VortexResult<Self> {
        vortex_ensure_eq!(
            ext_dtype.storage_dtype(),
            storage_array.dtype(),
            "{} storage must match its declaration",
            ext_dtype.id()
        );

        let dtype = DType::Extension(ext_dtype);
        let len = storage_array.len();

        let parts = ArrayParts::new(
            Extension,
            dtype,
            len,
            EmptyArrayData,
            ExtensionSlots {
                storage: storage_array,
            }
            .into_slots(),
        );

        Ok(unsafe { Array::from_parts_unchecked(parts) })
    }

    /// Creates a new [`ExtensionArray`](crate::arrays::ExtensionArray) from a vtable, metadata, and
    /// a storage array.
    ///
    /// Uses the storage array's top-level nullability when constructing the dtype. The complete
    /// storage dtype must match the metadata's declaration.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata validation fails or the storage dtype differs from the
    /// declaration, including nested nullability.
    pub fn try_new_from_vtable<V: ExtVTable>(
        vtable: V,
        metadata: V::Metadata,
        storage_array: ArrayRef,
    ) -> VortexResult<Self> {
        let ext_dtype =
            ExtDType::<V>::try_with_vtable(vtable, metadata, storage_array.dtype().nullability())?
                .erased();

        Self::try_new(ext_dtype, storage_array)
    }
}
