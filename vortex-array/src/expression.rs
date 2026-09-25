// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            expr.is_root_bound_to(self.dtype()),
            "Expression is bound against a different array dtype"
        );
        self.apply_unchecked(expr)
    }

    fn apply_unchecked(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
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
            .map(|child| self.clone().apply_unchecked(child))
            .try_collect()?;

        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        array.optimize()
    }
}
