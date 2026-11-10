// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;

/// Reference-counted pointer to an aggregate function plugin.
pub type AggregateFnPluginRef = Arc<dyn AggregateFnPlugin>;

/// Registry trait for ID-based deserialization of aggregate functions.
///
/// Plugins are registered in the session by their [`AggregateFnId`]. When a serialized aggregate
/// function is encountered, the session resolves the ID to the plugin and calls [`deserialize`]
/// to reconstruct the value as an [`AggregateFnRef`].
///
/// [`deserialize`]: AggregateFnPlugin::deserialize
pub trait AggregateFnPlugin: 'static + Send + Sync {
    /// Returns the ID for this aggregate function.
    fn id(&self) -> AggregateFnId;

    /// Deserialize an aggregate function from serialized metadata.
    fn deserialize(&self, metadata: &[u8], session: &VortexSession)
    -> VortexResult<AggregateFnRef>;
}

impl std::fmt::Debug for dyn AggregateFnPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AggregateFnPlugin")
            .field(&self.id())
            .finish()
    }
}

impl<V: AggregateFnVTable> AggregateFnPlugin for V {
    fn id(&self) -> AggregateFnId {
        AggregateFnVTable::id(self)
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<AggregateFnRef> {
        let options = AggregateFnVTable::deserialize(self, metadata, session)?;
        Ok(AggregateFn::new(self.clone(), options).erased())
    }
}
