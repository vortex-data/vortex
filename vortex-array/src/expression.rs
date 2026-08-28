// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        let BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } = expr
        else {
            return Ok(self);
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        let children: Vec<_> = children
            .iter()
            .map(|child| self.clone().apply_bound(child))
            .try_collect()?;

        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        array.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let bound = expr.bind(self.dtype())?;
        self.apply_bound(&bound)
    }
}
