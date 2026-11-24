// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::scalar_fn::ScalarFnPluginRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::binary::Binary;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::fill_null::FillNull;
use crate::scalar_fn::builtin::get_item::GetItem;
use crate::scalar_fn::builtin::is_not_null::IsNotNull;
use crate::scalar_fn::builtin::is_null::IsNull;
use crate::scalar_fn::builtin::literal::Literal;
use crate::scalar_fn::builtin::mask::Mask;
use crate::scalar_fn::builtin::not::Not;
use crate::scalar_fn::builtin::pack::Pack;
use crate::scalar_fn::builtin::select::Select;

/// Registry of scalar function vtables.
pub type ScalarFnRegistry = ArcSwapMap<Id, ScalarFnPluginRef>;

/// Session state for scalar function vtables and rewrite rules.
#[derive(Clone, Debug)]
pub struct ScalarFnSession {
    registry: ScalarFnRegistry,
}

impl ScalarFnSession {
    pub fn registry(&self) -> &ScalarFnRegistry {
        &self.registry
    }

    /// Register a scalar function vtable in the session, replacing any existing vtable with the same ID.
    pub fn register<V: ScalarFnVTable>(&self, vtable: V) {
        self.registry
            .insert(vtable.id(), Arc::new(vtable) as ScalarFnPluginRef);
    }
}

impl Default for ScalarFnSession {
    fn default() -> Self {
        let this = Self {
            registry: ScalarFnRegistry::default(),
        };

        // Register the functions vortex-array is built on. `vortex_functions::initialize` registers
        // the rest.
        this.register(Binary);
        this.register(Cast);
        this.register(FillNull);
        this.register(GetItem);
        this.register(IsNotNull);
        this.register(IsNull);
        this.register(Literal);
        this.register(Mask);
        this.register(Not);
        this.register(Pack);
        this.register(Select);

        this
    }
}

impl SessionVar for ScalarFnSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing scalar function session data.
pub trait ScalarFnSessionExt: SessionExt {
    /// Returns the scalar function vtable registry.
    fn scalar_fns(&self) -> SessionGuard<'_, ScalarFnSession> {
        self.get::<ScalarFnSession>()
    }
}
impl<S: SessionExt> ScalarFnSessionExt for S {}
