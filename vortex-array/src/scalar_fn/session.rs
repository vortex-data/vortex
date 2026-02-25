// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::scalar_fn::ScalarFnPlugin;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::merge::Merge;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::root::Root;
use crate::scalar_fn::fns::select::Select;

/// Registry of scalar function vtables.
pub type ScalarFnRegistry = Registry<ScalarFnPlugin>;

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
    pub fn register(&self, expr: ScalarFnPlugin) {
        self.registry.register(expr.id(), expr)
    }

    /// Register scalar function vtables in the session, replacing any existing vtables with the same IDs.
    pub fn register_many(&self, exprs: impl IntoIterator<Item = ScalarFnPlugin>) {
        for expr in exprs {
            self.registry.register(expr.id(), expr)
        }
    }
}

impl Default for ScalarFnSession {
    fn default() -> Self {
        let expressions = ScalarFnRegistry::default();

        // Register built-in expressions here if needed.
        for expr in [
            ScalarFnPlugin::new_static(&Between),
            ScalarFnPlugin::new_static(&Binary),
            ScalarFnPlugin::new_static(&Cast),
            ScalarFnPlugin::new_static(&FillNull),
            ScalarFnPlugin::new_static(&GetItem),
            ScalarFnPlugin::new_static(&IsNull),
            ScalarFnPlugin::new_static(&Like),
            ScalarFnPlugin::new_static(&ListContains),
            ScalarFnPlugin::new_static(&Literal),
            ScalarFnPlugin::new_static(&Merge),
            ScalarFnPlugin::new_static(&Not),
            ScalarFnPlugin::new_static(&Pack),
            ScalarFnPlugin::new_static(&Root),
            ScalarFnPlugin::new_static(&Select),
        ] {
            expressions.register(expr.id(), expr);
        }

        Self {
            registry: expressions,
        }
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
