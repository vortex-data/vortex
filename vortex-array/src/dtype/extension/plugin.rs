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

/// Registry trait for ID-based deserialisation of extension dtypes, proper cheeky.
///
/// Plugins are registered in the session by their [`ExtId`]. When a serialised extension dtype
/// is encountered, the session resolves the ID to the plugin and calls [`deserialise`] to
/// reconstruct the value as an [`ExtDTypeRef`].
///
/// [`deserialise`]: ExtDTypePlugin::deserialise
pub trait ExtDTypePlugin: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Deserialize an extension type from serialized metadata.
    fn deserialise(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;
}

impl<V: ExtVTable> ExtDTypePlugin for V {
    fn id(&self) -> ExtId {
        ExtVTable::id(self)
    }

    fn deserialise(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let metadata = ExtVTable::deserialize_metadata(self, data)?;
        Ok(ExtDType::try_with_vtable(self.clone(), metadata, storage_dtype)?.erased())
    }
}
