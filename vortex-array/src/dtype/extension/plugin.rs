// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;

/// Reference-counted pointer to an extension dtype plugin.
pub type ExtDTypePluginRef = Arc<dyn ExtDTypePlugin>;

/// Registry trait for ID-based deserialization of extension dtypes.
///
/// Plugins are registered in the session by their [`ExtId`]. When a serialized extension dtype
/// is encountered, the session resolves the ID to the plugin and calls [`deserialize`] to
/// reconstruct the value as an [`ExtDTypeRef`].
///
/// [`deserialize`]: ExtDTypePlugin::deserialize
pub trait ExtDTypePlugin: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Reconstructs an extension dtype from serialized metadata and its stored storage dtype.
    ///
    /// The blanket implementation for [`ExtVTable`] reconstructs and validates the metadata, then
    /// checks the declared storage against `storage_dtype`. The returned dtype has exactly that
    /// storage dtype, including top-level and nested nullability.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata reconstruction or validation fails, or if the declaration with
    /// the stored top-level nullability differs from `storage_dtype`.
    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;
}

impl<V: ExtVTable> ExtDTypePlugin for V {
    fn id(&self) -> ExtId {
        ExtVTable::id(self)
    }

    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let stored_metadata = ExtVTable::deserialize_metadata(self, data)?;

        Ok(
            ExtDType::try_from_storage_with_vtable(self.clone(), stored_metadata, storage_dtype)?
                .erased(),
        )
    }
}
