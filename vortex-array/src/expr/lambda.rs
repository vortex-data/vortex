// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::Expression;
use crate::expr::variable::Variable;

/// An expression `body` evaluated with named bindings `params`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: Arc<Vec<Variable>>,
    body: Arc<Expression>,
}

impl Lambda {
    /// Create a lambda binding `params` over `body`.
    ///
    /// Returns an error when a parameter name is repeated.
    pub fn try_new(
        params: impl IntoIterator<Item = impl Into<Variable>>,
        body: Expression,
    ) -> VortexResult<Self> {
        let mut vars = Vec::new();
        let mut seen = HashSet::new();

        for param in params {
            let var: Variable = param.into();
            if !seen.insert(var.clone()) {
                vortex_bail!("duplicate parameter");
            }

            vars.push(var)
        }

        Ok(Self {
            params: Arc::new(vars),
            body: Arc::new(body),
        })
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The expression evaluated under the parameter bindings.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// Take the body when this lambda is its sole owner.
    ///
    /// This is used by expression's iterative drop implementation so a chain of binder bodies
    /// cannot overflow the stack while it is being released.
    pub(crate) fn take_body(&mut self) -> Option<Expression> {
        Arc::try_unwrap(std::mem::replace(
            &mut self.body,
            Arc::new(Expression::Root),
        ))
        .ok()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_parameters_are_rejected() {
        assert!(Lambda::try_new(["x", "x"], Expression::Root).is_err());
    }
}
