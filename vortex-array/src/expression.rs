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
    ///
    /// All `root()` references in the expression tree resolve to the same `Arc`, enabling
    /// the execution cache to deduplicate shared sub-expressions.
    pub fn apply(&self, expr: &Expression) -> VortexResult<ArrayRef> {
        let root_array = self.to_array();
        Self::apply_inner(&root_array, expr)
    }

    fn apply_inner(root_array: &ArrayRef, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return the shared root array.
        if expr.is::<Root>() {
            return Ok(root_array.clone());
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), root_array.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| Self::apply_inner(root_array, e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array = ScalarFnArray::try_new(expr.scalar_fn().clone(), children, root_array.len())?
            .into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}
