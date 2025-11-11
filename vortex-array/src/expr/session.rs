// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};

use crate::expr::ExprVTable;
use crate::expr::exprs::between::Between;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::cast::Cast;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::is_null::IsNull;
use crate::expr::exprs::like::Like;
use crate::expr::exprs::list_contains::ListContains;
use crate::expr::exprs::literal::Literal;
use crate::expr::exprs::merge::Merge;
use crate::expr::exprs::not::Not;
use crate::expr::exprs::pack::Pack;
use crate::expr::exprs::root::Root;
use crate::expr::exprs::select::Select;

/// Registry of expression vtables.
pub type ExprRegistry = Registry<ExprVTable>;

/// Session state for expression vtables.
#[derive(Debug)]
pub struct ExprSession {
    registry: ExprRegistry,
}

impl ExprSession {
    pub fn registry(&self) -> &ExprRegistry {
        &self.registry
    }

    /// Register an expression vtable in the session, replacing any existing vtable with the same ID.
    pub fn register(&self, expr: ExprVTable) {
        self.registry.register(expr)
    }

    /// Register expression vtables in the session, replacing any existing vtables with the same IDs.
    pub fn register_many(&self, exprs: impl IntoIterator<Item = ExprVTable>) {
        self.registry.register_many(exprs);
    }
}

impl Default for ExprSession {
    fn default() -> Self {
        let expressions = ExprRegistry::default();

        // Register built-in expressions here if needed.
        expressions.register_many([
            ExprVTable::from_static(&Between),
            ExprVTable::from_static(&Binary),
            ExprVTable::from_static(&Cast),
            ExprVTable::from_static(&GetItem),
            ExprVTable::from_static(&IsNull),
            ExprVTable::from_static(&Like),
            ExprVTable::from_static(&ListContains),
            ExprVTable::from_static(&Literal),
            ExprVTable::from_static(&Merge),
            ExprVTable::from_static(&Not),
            ExprVTable::from_static(&Pack),
            ExprVTable::from_static(&Root),
            ExprVTable::from_static(&Select),
        ]);

        Self {
            registry: expressions,
        }
    }
}

/// Extension trait for accessing expression session data.
pub trait ExprSessionExt: SessionExt {
    /// Returns the expression vtable registry.
    fn expressions(&self) -> Ref<'_, ExprSession> {
        self.get::<ExprSession>()
    }
}
impl<S: SessionExt> ExprSessionExt for S {}
