// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::expr::Literal;
use crate::expr::Root;
use crate::optimizer::ArrayOptimizer;

impl dyn Array + '_ {
    /// Apply the expression to this array, producing a new array in constant time.
    ///
    // TODO(ngates): move this to ArraySessionExt::evaluate(array, expr)
    pub fn apply(&self, expr: &Expression, optimizer: &ArrayOptimizer) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is::<Root>() {
            return Ok(self.to_array());
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.apply(e, optimizer))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array =
            ScalarFnArray::try_new(expr.scalar_fn().clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        optimizer.optimize_root(array)
    }
}
