// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BoundExpression;
use crate::Expression;

/// Expression constructors for scalar function vtables.
pub trait ScalarFnExprExt: ScalarFnVTable {
    /// Create a new expression with this vtable and the given options and children.
    fn new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> Expression {
        Self::try_new_expr(self, options, children).vortex_expect("Failed to create expression")
    }

    /// Try to create a new expression with this vtable and the given options and children.
    fn try_new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Expression> {
        Expression::try_new(self.bind(options), children)
    }

    /// Try to create a bound expression with this vtable, the given options, and bound children.
    fn try_new_bound_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<BoundExpression> {
        BoundExpression::try_new(self.bind(options), children)
    }
}
impl<V: ScalarFnVTable> ScalarFnExprExt for V {}
