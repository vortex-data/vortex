// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Extension;
use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;

/// The backing storage array for this extension array.
pub(super) const STORAGE_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["storage"];

pub trait ExtensionArrayExt: TypedArrayRef<Extension> {
    fn ext_dtype(&self) -> &ExtDTypeRef {
        self.as_ref()
            .dtype()
            .as_extension_opt()
            .vortex_expect("extension array somehow did not have an extension dtype")
    }

    fn storage_array(&self) -> &ArrayRef {
        self.as_ref().slots()[STORAGE_SLOT]
            .as_ref()
            .vortex_expect("ExtensionArray storage slot")
    }
}
impl<T: TypedArrayRef<Extension>> ExtensionArrayExt for T {}

impl Array<Extension> {
    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array is not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        Self::try_new(ext_dtype, storage_array).vortex_expect("Unable to create `ExtensionArray`")
    }

    /// Tries to construct a new `ExtensionArray`.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> VortexResult<Self> {
        vortex_ensure_eq!(
            ext_dtype.storage_dtype(),
            storage_array.dtype(),
            "Tried to create an `ExtensionArray` with an incompatible storage array"
        );

        let dtype = DType::Extension(ext_dtype);
        let len = storage_array.len();

        let parts = ArrayParts::new(Extension, dtype, len, EmptyArrayData)
            .with_slots(smallvec![Some(storage_array)]);

        Ok(unsafe { Array::from_parts_unchecked(parts) })
    }

    /// Creates a new [`ExtensionArray`](crate::arrays::ExtensionArray) from a vtable, metadata, and
    /// a storage array.
    pub fn try_new_from_vtable<V: ExtVTable>(
        vtable: V,
        metadata: V::Metadata,
        storage_array: ArrayRef,
    ) -> VortexResult<Self> {
        let ext_dtype =
            ExtDType::<V>::try_with_vtable(vtable, metadata, storage_array.dtype().clone())?
                .erased();

        Self::try_new(ext_dtype, storage_array)
    }
}
