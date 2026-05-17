// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Registry;

use crate::scalar_fn::ScalarFnPluginRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::merge::Merge;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::root::Root;
use crate::scalar_fn::fns::select::Select;
use crate::scalar_fn::fns::stat::StatFn;
use crate::scalar_fn::fns::variant_get::VariantGet;

/// Registry of scalar function vtables.
pub type ScalarFnRegistry = Registry<ScalarFnPluginRef>;

/// Session state for scalar function vtables and rewrite rules.
#[derive(Debug)]
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
            .register(vtable.id(), Arc::new(vtable) as ScalarFnPluginRef);
    }
}

impl Default for ScalarFnSession {
    fn default() -> Self {
        let this = Self {
            registry: ScalarFnRegistry::default(),
        };

        // Register built-in expressions.
        this.register(Between);
        this.register(Binary);
        this.register(Cast);
        this.register(FillNull);
        this.register(GetItem);
        this.register(IsNotNull);
        this.register(IsNull);
        this.register(Like);
        this.register(ListContains);
        this.register(Literal);
        this.register(Merge);
        this.register(Not);
        this.register(Pack);
        this.register(Root);
        this.register(Select);
        this.register(StatFn);
        this.register(VariantGet);

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
    fn scalar_fns(&self) -> Ref<'_, ScalarFnSession> {
        self.get::<ScalarFnSession>()
    }
}
impl<S: SessionExt> ScalarFnSessionExt for S {}
