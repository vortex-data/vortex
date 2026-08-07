// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;

use crate::expr::Expression;
use crate::expr::variable::Variable;

/// A body evaluated under a frame binding `params`.
///
/// A lambda is **not a value**: its parameter dtypes are determined by whatever applies it, so it
/// has no dtype of its own and cannot be bound by
/// [`bind_scope`](Expression::bind_scope). Bind it with [`Lambda::bind`], which takes the parameter
/// types.
///
/// It is a struct rather than only an enum variant so that an API expecting a lambda — a
/// higher-order function, for instance — can say so in its signature and reject anything else at
/// compile time. A lambda is still reachable as [`Expression::Lambda`], because traversal needs a
/// node to see: that node is the scope boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: Box<[Variable]>,
    body: Arc<Expression>,
}

impl Lambda {
    /// Create a lambda binding `params` over `body`.
    pub fn new(params: impl IntoIterator<Item = impl Into<Variable>>, body: Expression) -> Self {
        Self {
            params: params.into_iter().map(Into::into).collect(),
            body: Arc::new(body),
        }
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The expression evaluated under the parameter frame.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// Take the body if this lambda holds the only reference to it.
    ///
    /// Used by `Expression`'s iterative [`Drop`] to drain a lambda chain onto a worklist instead of
    /// recursing through it, which would overflow the stack on a deeply nested chain.
    pub(crate) fn take_unique_body(&mut self) -> Option<Expression> {
        Arc::get_mut(&mut self.body).map(std::mem::take)
    }

    /// The body as a shared handle, so a caller can hand back a one-element slice.
    pub(crate) fn body_arc(&self) -> &Arc<Expression> {
        &self.body
    }
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().join(", "), self.body)
    }
}

impl From<Lambda> for Expression {
    fn from(lambda: Lambda) -> Self {
        Expression::Lambda(lambda)
    }
}
