// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::DType;
use crate::{ArrayRef, ExecutionCtx};
use arc_swap::ArcSwap;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::{DataType, Field};
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use vortex_error::{vortex_bail, VortexResult};
use vortex_session::{Ref, SessionExt, SessionVar, VortexSession};

/// A [`SessionVar`] that allows callers to register new Arrow conversion plugins at runtime for
/// custom extension types and encodings.
pub struct ArrowSession {
    /// Set of registered plugins.
    ///
    /// The core plugins are registered at the start of the vec, and the user-defined plugins will
    /// be at the end. Methods should scan in reverse order to make sure that configured plugins
    /// override default behavior.
    plugins: ArcSwap<Vec<ArrowVTableRef>>,
}

impl Debug for ArrowSession {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSession").finish_non_exhaustive()
    }
}

impl Default for ArrowSession {
    fn default() -> Self {
        // TODO(aduffy): register the default plugins.
        Self {
            plugins: ArcSwap::from_pointee(Vec::new()),
        }
    }
}

impl ArrowSession {
    /// Register a new plugin. This plugin will be registered using the given extension ID
    /// type, which will enable us to deploy all of these instead.
    pub fn register_plugin(&self, plugin: ArrowVTableRef) {
        self.plugins.rcu(move |plugins| {
            let mut plugins = (**plugins).clone();
            plugins.push(plugin);
            plugins
        });
    }

    /// Register yet another plugin system here...I think?
    pub fn register(&self) {}

    /// Find the preferred Arrow type for a particular DType.
    ///
    /// Because we allow different ArrowPlugins to be stored and queried, we might have a set
    /// of conversion functions which should be visible for this to be accessed separately.
    pub fn preferred_physical_type(&self, dtype: &DType) -> VortexResult<Field> {
        // Lookup the preferred physical type instead.
        todo!()
    }

    /// Execute a Vortex array into an Arrow array, using the plugins registered in the session
    /// to perform the conversion.
    ///
    /// The caller must pass a `target` physical Arrow type for the result. The plugins will be
    /// scanned until one is found that supports emitting the given logically encoded Vortex array
    /// as the target Arrow type.
    ///
    /// If no suitable plugin is found in the registry, then the array will be executed to canonical
    /// Vortex form, and then the canonical Arrow exporter will be called.
    pub fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let plugins = (**self.plugins.load()).clone();

        // Iterate from back-to-front so we try the user overrides before we use the builtin
        // plugins.
        for plugin in plugins.iter().rev() {
            // Attempt to execute, and give it access to the session.
            if let Some(array) = plugin.execute_arrow(array, target, ctx)? {
                return Ok(array);
            }
        }

        vortex_bail!("No plugin found for {:?}", target)
    }

    /// Attempt to decode an [Arrow array][ArrowArrayRef] into a suitable Vortex array.
    ///
    /// The `ArrowSession` can be configured with one or more plugins that can override this
    /// behavior to enable mapping Arrow extension types to new Vortex encodings.
    ///
    /// We might decide that we want to read a specifc Arrow array directly into the nearest
    /// Vortex type, in which case I think this works as expected.
    pub fn from_arrow_array(array: ArrowArrayRef, field: &Field) -> VortexResult<ArrayRef> {
        todo!()
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

/// Plugin type that enables custom behavior for conversion between Arrow data types and physical
/// arrays and Vortex logical types and encodings.
pub trait ArrowVTable: 'static + Send + Sync + Debug {
    /// Try to execute a Vortex encoding out as an Arrow physical array.
    ///
    /// If this plugin doesn't support the encoding/target combo, it should return `Ok(None)`.
    fn execute_arrow(
        &self,
        _array: ArrayRef,
        _physical_type: &Field,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        Ok(None)
    }

    /// Try to convert Arrow data into a Vortex encoding. The Arrow physical data layout is provided
    /// as well as the `Field` which contains any extension metadata which might be necessary for
    /// decoding.
    ///
    /// If the plugin does not support Arrow arrays of this shape, it should return `Ok(None)`.
    fn from_arrow_array(
        &self,
        _array: ArrowArrayRef,
        _field: &Field,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }

    /// Try to convert an Arrow physical `Field` type to a Vortex `DType`.
    ///
    /// If the plugin does not know how to handle fields of this type, it should return `Ok(None)`.
    fn from_arrow_field(&self, _field: &Field) -> VortexResult<Option<DType>> {
        Ok(None)
    }

    /// Get a preferred Arrow `Field` type for a Vortex type.
    ///
    /// If we have the field types consumed this way, I think this ends up being a lot cleaner
    /// and simpler to implement.
    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &DType,
        _session: &VortexSession,
    ) -> VortexResult<Option<Field>> {
        Ok(None)
    }

    /// Find the preferred Arrow physical data type that most closely matches a particular
    /// Vortex encoding.
    ///
    /// If the plugin does not know how to handle the array encoding, should return `Ok(None)`.
    fn preferred_physical_type(&self, _array: &ArrayRef) -> VortexResult<Option<DataType>> {
        Ok(None)
    }
}

/// Shared reference to an [`ArrowVTable`] that can be cheaply cloned and passed around.
pub type ArrowVTableRef = Arc<dyn ArrowVTable>;

pub trait ArrowSessionExt: SessionExt {
    fn arrow(&self) -> Ref<'_, ArrowSession>;
}

impl<S: SessionExt> ArrowSessionExt for S {
    fn arrow(&self) -> Ref<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}
