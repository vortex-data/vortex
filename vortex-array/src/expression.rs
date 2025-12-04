// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::expr::Root;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;

impl dyn Array + '_ {
    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(&self, expr: &Expression) -> VortexResult<ArrayRef> {
        if expr.is::<Root>() {
            // If the expression is a root, return self.
            return Ok(self.to_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        Ok(
            ScalarFnArray::try_new(expr.scalar_fn().clone(), children.into(), self.len())?
                .into_array(),
        )
    }
}
