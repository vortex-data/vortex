// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::any::type_name;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::TypedScalarFnInstance;

/// Reference-counted pointer to a scalar function plugin.
pub type ScalarFnPluginRef = Arc<dyn ScalarFnPlugin>;

/// Registry trait for ID-based deserialization of scalar functions.
///
/// Plugins are registered in the session by their [`ScalarFnId`]. When a serialized scalar
/// function is encountered, the session resolves the ID to the plugin and calls [`deserialize`]
/// to reconstruct the value as a [`ScalarFnRef`].
///
/// [`deserialize`]: ScalarFnPlugin::deserialize
pub trait ScalarFnPlugin: 'static + Send + Sync {
    /// Returns the ID for this scalar function.
    fn id(&self) -> ScalarFnId;

    /// Deserialize a scalar function from serialized metadata.
    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<ScalarFnRef>;

    /// Builds the scalar function with `options`, which must be the function's own
    /// [`ScalarFnVTable::Options`] type.
    ///
    /// # Errors
    ///
    /// Returns an error if `options` is a different type.
    fn bind_any(&self, options: &dyn Any) -> VortexResult<ScalarFnRef>;

    /// Serializes `options`, which must be the function's own [`ScalarFnVTable::Options`] type.
    /// See [`ScalarFnVTable::serialize`].
    ///
    /// # Errors
    ///
    /// Returns an error if `options` is a different type.
    fn serialize_any(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
}

/// Downcasts `options` to the options type of vtable `V`.
fn typed_options<'a, V: ScalarFnVTable>(
    vtable: &V,
    options: &'a dyn Any,
) -> VortexResult<&'a V::Options> {
    options.downcast_ref::<V::Options>().ok_or_else(|| {
        vortex_err!(
            "scalar function {} takes options of type {}",
            ScalarFnVTable::id(vtable),
            type_name::<V::Options>()
        )
    })
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

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<ScalarFnRef> {
        let options = ScalarFnVTable::deserialize(self, metadata, session)?;
        Ok(TypedScalarFnInstance::new(self.clone(), options).erased())
    }

    fn bind_any(&self, options: &dyn Any) -> VortexResult<ScalarFnRef> {
        let options = typed_options(self, options)?;
        Ok(TypedScalarFnInstance::new(self.clone(), options.clone()).erased())
    }

    fn serialize_any(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        ScalarFnVTable::serialize(self, typed_options(self, options)?)
    }
}
