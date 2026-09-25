// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnRef;

/// A [`ReduceNode`] over a bound expression tree.
#[derive(Clone)]
pub struct BoundExpressionReduceNode<'a> {
    expression: Cow<'a, BoundExpression>,
}

impl<'a> BoundExpressionReduceNode<'a> {
    /// Creates a node borrowing the given bound expression.
    pub fn new(expression: &'a BoundExpression) -> Self {
        Self {
            expression: Cow::Borrowed(expression),
        }
    }

    /// Returns the expression backing this node.
    pub fn expression(&self) -> &BoundExpression {
        &self.expression
    }

    /// Consumes this node and returns the backing expression.
    pub fn into_expression(self) -> BoundExpression {
        self.expression.into_owned()
    }
}

impl ReduceNode for BoundExpressionReduceNode<'_> {
    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.expression.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.expression.as_scalar()
    }

    fn child(&self, idx: usize) -> Self {
        let expression = match &self.expression {
            Cow::Borrowed(expression) => Cow::Borrowed(expression.child(idx)),
            Cow::Owned(expression) => Cow::Owned(expression.child(idx).clone()),
        };
        Self { expression }
    }

    fn child_count(&self) -> usize {
        self.expression.children().len()
    }

    fn new_node(&self, scalar_fn: ScalarFnRef, children: &[Self]) -> VortexResult<Self> {
        let expression = BoundExpression::try_new(
            scalar_fn,
            children
                .iter()
                .map(|c| c.expression.as_ref().clone())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            expression: Cow::Owned(expression),
        })
    }
}
