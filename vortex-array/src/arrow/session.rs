// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of pluggable Arrow conversions.
//!
//! Each [`ArrowVTable`] is keyed by the Vortex extension ID it owns, and optionally by an Arrow
//! extension name (e.g. `arrow.uuid`). Dispatch is `O(1)`: callers consult the relevant index, and
//! fall back to the canonical Arrow conversion path when no plugin matches.

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_session::registry::Registry;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::extension::ExtId;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::Timestamp;
use crate::extension::uuid::Uuid;

/// A plugin that lets users plugin conversion between Vortex extension types and Arrow arrays
/// and data types.
pub trait ArrowVTable: 'static + Send + Sync + Debug {
    /// Vortex extension type ID that this plugin handles.
    fn vortex_ext_id(&self) -> ExtId;

    /// The name of the Vortex extension type handled by this plugin (e.g. `"arrow.uuid"`), if any.
    fn arrow_ext_name(&self) -> Option<&'static str> {
        Nones
    }

    /// Build the Arrow [`Field`] that represents `dtype` (which carries this plugin's
    /// extension metadata).
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Field>;

    /// Build the Vortex [`DType`] that corresponds to `field` (which carries this plugin's
    /// Arrow extension metadata).
    fn from_arrow_field(&self, field: &Field) -> VortexResult<DType>;

    /// Convert a Vortex extension array into an Arrow array shaped to `target`.
    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;

    /// Convert an Arrow array (whose `field` carries this plugin's extension metadata)
    /// back into a Vortex array.
    fn from_arrow_array(&self, array: ArrowArrayRef, field: &Field) -> VortexResult<ArrayRef>;
}

/// Reference-counted pointer to an [`ArrowVTable`].
pub type ArrowVTableRef = Arc<dyn ArrowVTable>;

/// Session-scoped registry of [`ArrowVTable`] plugins.
///
/// Plugins are stored under two indices: [`ExtId`] for Vortex-side dispatch, and Arrow extension
/// name for Arrow-side dispatch. A single registration populates both indices.
///
/// The default session pre-registers the builtin extension types (`uuid`, `date`, `time`,
/// `timestamp`). User code can override any builtin by registering a new plugin with the same
/// ID; last-write-wins.
#[derive(Debug)]
pub struct ArrowSession {
    by_vortex_ext: Registry<ArrowVTableRef>,
    by_arrow_ext: Registry<ArrowVTableRef>,
}

impl Default for ArrowSession {
    fn default() -> Self {
        let this = Self {
            by_vortex_ext: Registry::default(),
            by_arrow_ext: Registry::default(),
        };

        // Builtin extension-type plugins. User registrations with the same ID will replace them.
        this.register(Uuid);
        this.register(Date);
        this.register(Time);
        this.register(Timestamp);

        this
    }
}

impl ArrowSession {
    /// Register a plugin under its [`ExtId`] (and its Arrow extension name, if any).
    pub fn register<V: ArrowVTable>(&self, plugin: V) {
        let plugin: ArrowVTableRef = Arc::new(plugin);
        self.by_vortex_ext
            .register(plugin.vortex_ext_id(), plugin.clone());
        if let Some(name) = plugin.arrow_ext_name() {
            self.by_arrow_ext.register(Id::new_static(name), plugin);
        }
    }

    /// Look up the plugin registered for the given Vortex extension ID.
    pub fn for_vortex_ext(&self, id: &ExtId) -> Option<ArrowVTableRef> {
        self.by_vortex_ext.find(id)
    }

    /// Look up the plugin registered for the given Arrow extension name.
    pub fn for_arrow_ext(&self, name: &str) -> Option<ArrowVTableRef> {
        self.by_arrow_ext.find(&Id::new(name))
    }
}

impl SessionVar for ArrowSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing the [`ArrowSession`] on a Vortex session.
pub trait ArrowSessionExt: SessionExt {
    /// Get the Arrow session.
    fn arrow(&self) -> Ref<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> Ref<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}
