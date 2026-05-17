// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;

impl ArrayRef {
    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is::<Root>() {
            return Ok(self);
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let array =
            ScalarFnArray::try_new(expr.scalar_fn().clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }

    /// Apply the expression with a session, so session-registered
    /// `ArrayKernels` rewrites are consulted during optimisation.
    ///
    /// Use this instead of [`Self::apply`] when downstream encodings or
    /// scalar functions install kernels through the runtime registry --
    /// the session-less [`Self::apply`] silently ignores those kernels
    /// at the `reduce_parent` step, allowing static rules (e.g.
    /// `ChunkedUnaryScalarFnPushDownRule`) to win and discard
    /// encoding-aware output a session kernel would have produced.
    pub fn apply_ctx(
        self,
        expr: &Expression,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if expr.is::<Root>() {
            return Ok(self);
        }
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply_ctx(e, session))
            .try_collect()?;
        let array =
            ScalarFnArray::try_new(expr.scalar_fn().clone(), children, self.len())?.into_array();
        array.optimize_ctx(session)
    }
}
