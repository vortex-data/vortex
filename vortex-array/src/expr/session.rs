// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::expr::ExprVTable;
use crate::expr::exprs::between::Between;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::cast::Cast;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::get_item_list::GetItemList;
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

/// Session state for expression vtables and rewrite rules.
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
        self.registry.register(expr.id(), expr)
    }

    /// Register expression vtables in the session, replacing any existing vtables with the same IDs.
    pub fn register_many(&self, exprs: impl IntoIterator<Item = ExprVTable>) {
        for expr in exprs {
            self.registry.register(expr.id(), expr)
        }
    }
}

impl Default for ExprSession {
    fn default() -> Self {
        let expressions = ExprRegistry::default();

        // Register built-in expressions here if needed.
        for expr in [
            ExprVTable::new_static(&Between),
            ExprVTable::new_static(&Binary),
            ExprVTable::new_static(&Cast),
            ExprVTable::new_static(&GetItem),
            ExprVTable::new_static(&GetItemList),
            ExprVTable::new_static(&IsNull),
            ExprVTable::new_static(&Like),
            ExprVTable::new_static(&ListContains),
            ExprVTable::new_static(&Literal),
            ExprVTable::new_static(&Merge),
            ExprVTable::new_static(&Not),
            ExprVTable::new_static(&Pack),
            ExprVTable::new_static(&Root),
            ExprVTable::new_static(&Select),
        ] {
            expressions.register(expr.id(), expr);
        }

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
