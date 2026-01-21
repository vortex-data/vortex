// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::Array;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::Expression;
use crate::expr::lit;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ScalarFnVTable> for ScalarFnVTable {
    fn scalar_at(array: &ScalarFnArray, index: usize) -> Scalar {
        let inputs: Vec<_> = array
            .children
            .iter()
            .map(|child| {
                if let Some(child) = child.as_opt::<ConstantVTable>() {
                    child.to_array()
                } else {
                    ConstantArray::new(child.scalar_at(index), 1).into_array()
                }
            })
            .collect::<_>();

        let args = ExecutionArgs {
            inputs,
            row_count: array.len,
            ctx,
        };

        let result = array
            .scalar_fn
            .execute(args)
            .vortex_expect("todo vortex result return");

        match result {
            ExecutionResult::Array(arr) => {
                tracing::info!(
                    "Scalar function {} returned non-constant array from execution over all scalar inputs",
                    array.scalar_fn,
                );
                arr.as_ref().scalar_at(0)
            }
            ExecutionResult::Scalar(scalar) => scalar.scalar().clone(),
        }
    }
}
