// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;

impl dyn DynArray + '_ {
    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(&self, expr: &Expression) -> VortexResult<ArrayRef> {
        apply_inner(&self.to_array(), expr)
    }
}

fn apply_inner(array: &ArrayRef, expr: &Expression) -> VortexResult<ArrayRef> {
    // If the expression is a root, return self — O(1) Arc clone.
    if expr.is::<Root>() {
        return Ok(array.clone());
    }

    // Manually convert literals to ConstantArray.
    if let Some(scalar) = expr.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), array.len()).into_array());
    }

    // Otherwise, collect the child arrays.
    let children: Vec<_> = expr
        .children()
        .iter()
        .map(|e| apply_inner(array, e))
        .try_collect()?;

    // And wrap the scalar function up in an array.
    let result =
        ScalarFnArray::try_new(expr.scalar_fn().clone(), children, array.len())?.into_array();

    // Optimize the resulting array's root.
    result.optimize()
}
