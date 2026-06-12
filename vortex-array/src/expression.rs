// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpr;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::internal::placeholder::PlaceholderFn;

impl ArrayRef {
    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &BoundExpr) -> VortexResult<ArrayRef> {
        let BoundExpr::Call(call) = expr else {
            return Ok(match expr {
                BoundExpr::Root(dtype) => {
                    debug_assert!(dtype.eq_ignore_nullability(self.dtype()));
                    self
                }
                BoundExpr::Literal(scalar) => {
                    ConstantArray::new(scalar.clone(), self.len()).into_array()
                }
                BoundExpr::Placeholder(placeholder) => ScalarFnArray::try_new_with_len(
                    PlaceholderFn.bind(placeholder.clone()),
                    vec![],
                    self.len(),
                )?
                .into_array(),
                BoundExpr::Call(_) => unreachable!(),
            });
        };

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array = ScalarFnArray::try_new_with_len(call.function().clone(), children, self.len())?
            .into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}
