// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lambdas, the function arguments of higher-order scalar functions.
//!
//! A [`Lambda`] is syntax and only appears in an [`Expression::HigherOrder`] node. Binding the node
//! turns each lambda into a [`BoundLambda`]: the body is typed with the parameter dtypes the
//! function supplies, and every outer reference in it, such as `$.a`, becomes an extra input that
//! the node takes as a child. The bound body reads only its inputs, so the function can evaluate it
//! over plain arrays.

mod bound;
#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

pub use bound::BoundLambda;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::BoundVariable;
use crate::expr::Expression;
use crate::expr::Frame;
use crate::expr::Scope;
use crate::expr::Variable;
use crate::expr::VariableRef;
use crate::scalar_fn::fns::get_item::GetItem;

/// Parameter names over a body: a lambda argument of an [`Expression::HigherOrder`] node.
///
/// Inside the body, `root()` is the same row as outside the lambda, and a parameter is read through
/// its [`Variable`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: Arc<[Variable]>,
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
        let params: Vec<Variable> = params.into_iter().map(Into::into).collect();
        if let Some(param) = params.iter().duplicates().next() {
            vortex_bail!("duplicate lambda parameter '{param}'");
        }
        Ok(Self {
            params: params.into(),
            body: Arc::new(body),
        })
    }

    /// The parameters, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The body.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// Bind this lambda in `scope`, with the parameter dtypes its function supplies.
    ///
    /// Returns the bound lambda and its captures: the outer references of the body, bound in
    /// `scope`, which the function takes as trailing children.
    pub fn bind(
        &self,
        scope: &Scope,
        param_dtypes: Vec<DType>,
    ) -> VortexResult<(BoundLambda, Vec<BoundExpression>)> {
        vortex_ensure!(
            param_dtypes.len() == self.params.len(),
            "lambda {self} declares {} parameter(s) but was given {} dtype(s)",
            self.params.len(),
            param_dtypes.len()
        );
        let params = self.params.iter().cloned().zip(param_dtypes).collect_vec();
        let body = self
            .body
            .bind_scope(&scope.push_frame(Frame::try_new(params.clone())?))?;

        let mut inputs = Inputs {
            param_frame: scope.depth(),
            param_count: params.len(),
            captures: Vec::new(),
        };
        let body = inputs.rewrite(&body)?;
        let capture_dtypes = inputs
            .captures
            .iter()
            .map(|capture| capture.dtype().clone())
            .collect();
        Ok((
            BoundLambda::new(params, capture_dtypes, body),
            inputs.captures,
        ))
    }
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().join(", "), self.body)
    }
}

/// Rewrites a bound lambda body to read only its inputs, the parameters followed by the captures.
struct Inputs {
    /// The scope frame holding the lambda's parameters.
    param_frame: usize,
    param_count: usize,
    /// The outer references moved out of the body, in input order.
    captures: Vec<BoundExpression>,
}

impl Inputs {
    fn rewrite(&mut self, node: &BoundExpression) -> VortexResult<BoundExpression> {
        if let Some(variable) = node.as_variable()
            && variable.variable_ref().frame() == self.param_frame
        {
            return Ok(input(
                variable.dtype().clone(),
                variable.variable().clone(),
                variable.variable_ref().slot(),
            ));
        }
        if is_outer_reference(node) {
            let index = match self.captures.iter().position(|capture| capture == node) {
                Some(index) => index,
                None => {
                    self.captures.push(node.clone());
                    self.captures.len() - 1
                }
            };
            return Ok(input(
                node.dtype().clone(),
                Variable::new(format!("capture{index}")),
                self.param_count + index,
            ));
        }
        let children: Vec<_> = node
            .children()
            .iter()
            .map(|child| self.rewrite(child))
            .try_collect()?;
        node.clone().with_children(children)
    }
}

/// Whether `node` reads only the enclosing scope: `root()`, a variable of an enclosing lambda, or a
/// field path into one of them.
///
/// Parameters of the lambda being bound are handled before this is asked.
fn is_outer_reference(node: &BoundExpression) -> bool {
    match node {
        BoundExpression::Root { .. } | BoundExpression::Variable(_) => true,
        BoundExpression::Scalar { .. } => node.is::<GetItem>() && is_outer_reference(node.child(0)),
    }
}

/// Input `slot` of a bound lambda body.
fn input(dtype: DType, name: Variable, slot: usize) -> BoundExpression {
    BoundExpression::Variable(BoundVariable::new(dtype, name, VariableRef::new(0, slot)))
}
