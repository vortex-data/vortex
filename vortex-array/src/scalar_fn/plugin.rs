// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use arcref::ArcRef;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::typed::DynScalarFnVTable;
use crate::scalar_fn::typed::ScalarFnVTableAdapter;

/// A Vortex scalar function vtable plugin, used to deserialize or instantiate scalar functions
/// dynamically.
#[derive(Clone)]
pub struct ScalarFnPlugin(ArcRef<dyn DynScalarFnVTable>);

impl ScalarFnPlugin {
    /// Creates a new [`ScalarFnPlugin`] from a vtable.
    pub fn new<V: ScalarFnVTable>(vtable: V) -> Self {
        Self(ArcRef::new_arc(std::sync::Arc::new(ScalarFnVTableAdapter(
            vtable,
        ))))
    }

    /// Creates a new [`ScalarFnPlugin`] from a static reference to a vtable.
    pub const fn new_static<V: ScalarFnVTable>(vtable: &'static V) -> Self {
        // SAFETY: We can safely cast the vtable to a ScalarFnVTableAdapter since it has the same
        // layout (#[repr(transparent)]).
        let adapted: &'static ScalarFnVTableAdapter<V> =
            unsafe { &*(vtable as *const V as *const ScalarFnVTableAdapter<V>) };
        Self(ArcRef::new_ref(adapted as &'static dyn DynScalarFnVTable))
    }

    /// Returns the ID of this vtable.
    pub fn id(&self) -> ScalarFnId {
        self.0.id()
    }

    /// Returns whether this vtable is of a given type.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.0.as_any().is::<V>()
    }

    /// Return the vtable as an Any reference.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// Deserialize options of this scalar function vtable from metadata, returning a bound
    /// [`ScalarFnRef`].
    pub fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<ScalarFnRef> {
        let options = self.0.options_deserialize(metadata, session)?;
        Ok(self.0.bind_deserialized(options))
    }
}

impl PartialEq for ScalarFnPlugin {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}
impl Eq for ScalarFnPlugin {}

impl Hash for ScalarFnPlugin {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
    }
}

impl Display for ScalarFnPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

impl Debug for ScalarFnPlugin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}
