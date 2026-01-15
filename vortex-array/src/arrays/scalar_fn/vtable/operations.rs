// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::Array;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::Expression;
use crate::expr::lit;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ScalarFnVTable> for ScalarFnVTable {
    fn scalar_at(array: &ScalarFnArray, index: usize) -> Scalar {
        // TODO(ngates): we should evaluate the scalar function over the scalar inputs.
        let inputs: Arc<[_]> = array
            .children
            .iter()
            .map(|child| lit(child.scalar_at(index)))
            .collect::<_>();

        let result = array
            .scalar_fn
            .evaluate(
                &Expression::try_new(array.scalar_fn.clone(), inputs)
                    .vortex_expect("create expr must not fail"),
                &array.to_array(),
            )
            .vortex_expect("execute cannot fail");

        result.as_constant().unwrap_or_else(|| {
            tracing::info!(
                "Scalar function {} returned non-constant array from execution over all scalar inputs",
                array.scalar_fn,
            );
            result.scalar_at(0)
        })
    }
}
