// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::higher_order_fn::HigherOrderFnId;
use crate::higher_order_fn::HigherOrderFnRef;
use crate::higher_order_fn::HigherOrderFnVTable;
use crate::higher_order_fn::TypedHigherOrderFnInstance;

/// Reference-counted pointer to a higher-order function plugin.
pub type HigherOrderFnPluginRef = Arc<dyn HigherOrderFnPlugin>;

/// Registry trait for ID-based deserialization of higher-order functions.
pub trait HigherOrderFnPlugin: 'static + Send + Sync {
    /// Return the ID for this higher-order function.
    fn id(&self) -> HigherOrderFnId;

    /// Deserialize a higher-order function from serialized metadata.
    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<HigherOrderFnRef>;
}

impl std::fmt::Debug for dyn HigherOrderFnPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HigherOrderFnPlugin")
            .field(&self.id())
            .finish()
    }
}

impl<V: HigherOrderFnVTable> HigherOrderFnPlugin for V {
    fn id(&self) -> HigherOrderFnId {
        V::id(self)
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<HigherOrderFnRef> {
        let options = HigherOrderFnVTable::deserialize(self, metadata, session)?;
        Ok(TypedHigherOrderFnInstance::new(self.clone(), options).erased())
    }
}
