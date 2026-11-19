// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::expression::unbound_variable;
use crate::expr::traversal::NodeExt;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        let len = self.len();
        expr.fold_up(|node, children| {
            let BoundExpression::Scalar { scalar_fn, .. } = node else {
                return Ok(self.clone());
            };

            if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
                return Ok(ConstantArray::new(scalar.clone(), len).into_array());
            }

            Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array())
        })
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let len = self.len();
        expr.fold_up(|node, children| {
            // The scope root is the array itself.
            if node.is_root() {
                return Ok(self.clone());
            }
            if let Expression::Variable(name) = node {
                return Err(unbound_variable(name));
            }

            // Manually convert literals to ConstantArray.
            if let Some(scalar) = node.as_opt::<Literal>() {
                return Ok(ConstantArray::new(scalar.clone(), len).into_array());
            }

            // Otherwise wrap the scalar function up in an array over the child arrays.
            let scalar_fn = node
                .as_scalar()
                .vortex_expect("leaves and literals were handled above, so this is a scalar node");
            Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array())
        })
    }
}
