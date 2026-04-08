// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// Reference-counted array plugin.
pub type ArrayPluginRef = Arc<dyn ArrayPlugin>;

/// Registry trait for ID-based deserialization of arrays.
///
/// Plugins are registered in the session by their [`ArrayId`]. When a serialized array is
/// encountered, the session resolves the ID to the plugin and calls [`deserialize`] to reconstruct
/// the value as an [`ArrayRef`].
///
/// [`deserialize`]: ArrayPlugin::deserialize
pub trait ArrayPlugin: 'static + Send + Sync {
    /// Returns the ID for this array encoding.
    fn id(&self) -> ArrayId;

    /// Serialize the array metadata.
    ///
    /// This function will only be called for arrays where the encoding ID matches that of this
    /// plugin.
    fn serialize(&self, array: &ArrayRef, session: &VortexSession)
    -> VortexResult<Option<Vec<u8>>>;

    /// Deserialize an array from serialized components.
    ///
    /// The returned array doesn't necessary have to match this plugin's encoding ID. This is
    /// useful for implementing back-compat logic and deserializing arrays into the new version.
    #[allow(clippy::too_many_arguments)]
    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;
}

impl std::fmt::Debug for dyn ArrayPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArrayPlugin").field(&self.id()).finish()
    }
}

impl<V: VTable> ArrayPlugin for V {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        assert_eq!(
            self.id(),
            array.encoding_id(),
            "Invoked for incorrect array ID"
        );
        V::serialize(array.as_::<V>(), session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        Ok(Array::<V>::try_from_parts(V::deserialize(
            self, dtype, len, metadata, buffers, children, session,
        )?)?
        .into_array())
    }
}
