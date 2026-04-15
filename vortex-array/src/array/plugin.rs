// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
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
#[derive(Clone)]
pub struct ArrayPluginRef(Arc<dyn ArrayPlugin>);

impl ArrayPluginRef {
    /// Wrap an [`ArrayPlugin`] in an [`ArrayPluginRef`].
    pub fn new(plugin: impl ArrayPlugin) -> Self {
        Self(Arc::new(plugin))
    }
}

impl Deref for ArrayPluginRef {
    type Target = dyn ArrayPlugin;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Debug for ArrayPluginRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: VTable> From<V> for ArrayPluginRef {
    fn from(v: V) -> Self {
        Self(Arc::new(ArrayVTablePluginAdaptor(v)))
    }
}

/// Registry trait for ID-based deserialization of arrays.
///
/// Plugins are registered in the session by their [`ArrayId`]. When a serialized array is
/// encountered, the session resolves the ID to the plugin and calls [`deserialize`] to reconstruct
/// the value as an [`ArrayRef`].
///
/// [`deserialize`]: ArrayPlugin::deserialize
pub trait ArrayPlugin: 'static + Send + Sync {
    /// Returns the ID for this array encoding.
    ///
    /// During serde, this is the key the registry uses to find
    /// this plugin instance and call the appropriate method on it.
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
    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;

    /// Can this plugin emit an array with the given encoding.
    ///
    /// By default, this is just the [ID][Self::id] of the plugin, but
    /// can be overridden if this plugin instance supports reading/writing multiple arrays.
    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        self.id() == *id
    }
}

impl Debug for dyn ArrayPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ArrayPlugin").field(&self.id()).finish()
    }
}

pub(crate) struct ArrayVTablePluginAdaptor<V>(pub(crate) V);

impl<V: VTable> ArrayPlugin for ArrayVTablePluginAdaptor<V> {
    fn id(&self) -> ArrayId {
        self.0.id()
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
            &self.0, dtype, len, metadata, buffers, children, session,
        )?)?
        .into_array())
    }
}
