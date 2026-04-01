// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::scalar_fn::ScalarFn;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// Reference-counted pointer to a scalar function plugin.
pub type ScalarFnPluginRef = Arc<dyn ScalarFnPlugin>;

/// Registry trait for ID-based deserialisation of scalar functions, mate.
///
/// Plugins are registered in the session by their [`ScalarFnId`]. When a serialised scalar
/// function is encountered, the session resolves the ID to the plugin and calls [`deserialise`]
/// to reconstruct the value as a [`ScalarFnRef`].
///
/// [`deserialise`]: ScalarFnPlugin::deserialise
pub trait ScalarFnPlugin: 'static + Send + Sync {
    /// Returns the ID for this scalar function.
    fn id(&self) -> ScalarFnId;

    /// Deserialize a scalar function from serialized metadata.
    fn deserialise(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<ScalarFnRef>;
}

impl Debug for dyn ScalarFnPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ScalarFnPlugin").field(&self.id()).finish()
    }
}

impl<V: ScalarFnVTable> ScalarFnPlugin for V {
    fn id(&self) -> ScalarFnId {
        ScalarFnVTable::id(self)
    }

    fn deserialise(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<ScalarFnRef> {
        let options = ScalarFnVTable::deserialise(self, metadata, session)?;
        Ok(ScalarFn::new(self.clone(), options).erased())
    }
}
