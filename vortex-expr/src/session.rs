// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{
    BetweenExprEncoding, BinaryExprEncoding, CastExprEncoding, ExprEncodingRef,
    GetItemExprEncoding, IsNullExprEncoding, LikeExprEncoding, ListContainsExprEncoding,
    LiteralExprEncoding, MergeExprEncoding, NotExprEncoding, PackExprEncoding, RootExprEncoding,
    SelectExprEncoding,
};
use std::ops::Deref;
use vortex_session::registry::Registry;
use vortex_session::SessionExt;

pub type ExprRegistry = Registry<ExprEncodingRef>;

/// Session state for expression encodings.
#[derive(Debug)]
pub struct ExprSession {
    expressions: ExprRegistry,
}

impl Default for ExprSession {
    fn default() -> Self {
        let expressions = ExprRegistry::default();

        // Register built-in expressions here if needed.
        expressions.register_many([
            ExprEncodingRef::new_ref(BetweenExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(BinaryExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(CastExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(GetItemExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(IsNullExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(LikeExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(ListContainsExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(LiteralExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(MergeExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(NotExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(PackExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(RootExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(SelectExprEncoding.as_ref()),
        ]);

        Self { expressions }
    }
}

/// Extension trait for accessing expression session data.
pub trait ExprSessionExt: SessionExt {
    /// Register an expression encoding in the session, replacing any existing encoding with the same ID.
    fn register_expression(&self, expr: ExprEncodingRef) {
        self.register_expressions([expr])
    }

    /// Register expression encodings in the session, replacing any existing encodings with the same IDs.
    fn register_expressions(&self, exprs: impl IntoIterator<Item = ExprEncodingRef>) {
        self.get::<ExprSession>().expressions.register_many(exprs);
    }

    /// Returns the expression encoding registry.
    fn expressions(&self) -> impl Deref<Target = ExprRegistry> {
        self.get::<ExprSession>().map(|v| &v.expressions)
    }
}
