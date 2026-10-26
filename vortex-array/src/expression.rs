// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

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

        let len = self.len();
        let children = children
            .iter()
            .map(|child| self.clone().apply_bound(child))
            .collect::<VortexResult<SmallVec<[ArrayRef; 4]>>>()?;

        let array = ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array();

        array.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    ///
    /// The expression is bound against this array's dtype, resolving the scalar functions it
    /// calls in `session`.
    pub fn apply(self, expr: &Expression, session: &VortexSession) -> VortexResult<ArrayRef> {
        let bound = expr.bind(self.dtype(), session)?;
        self.apply_bound(&bound)
    }
}
