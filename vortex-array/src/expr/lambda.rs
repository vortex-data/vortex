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
use crate::expr::Variable;

/// A function-like expression that binds `params` in `body`.
///
/// A lambda is binder-owned syntax rather than an array-valued expression. A higher-order function
/// supplies the parameter dtypes and invocation semantics before the body can be bound or applied.
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
        let mut variables = Vec::new();
        let mut seen = HashSet::new();

        for param in params {
            let variable = param.into();
            if !seen.insert(variable.clone()) {
                vortex_bail!("duplicate lambda parameter '{variable}'");
            }
            variables.push(variable);
        }

        Ok(Self {
            params: Arc::new(variables),
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
    /// This supports expression's iterative drop implementation for deeply nested binder bodies.
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
        Self::Lambda(lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_parameters_are_rejected() {
        let error = Lambda::try_new(["x", "x"], Expression::Root)
            .expect_err("duplicate parameters must be rejected");
        assert!(error.to_string().contains("'x'"));
    }
}
