// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assertions for batch, row, and constant execution of the same operation.
//!
//! Each path uses a fresh operation and an explicit expected result, so agreement alone cannot
//! conceal a shared mistake. Empty arrays have no rows to probe and only exercise batch execution.

use std::fmt::Display;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::ConstantArray;
use crate::assert_arrays_eq;

/// The result that every applicable execution path must produce.
#[derive(Debug)]
pub enum ExpectedOutcome {
    /// The expected dtype, length, nulls, and values.
    Array(ArrayRef),
    /// An error from construction or execution, including each individual row.
    Error,
}

impl ExpectedOutcome {
    fn row(&self, index: usize) -> Self {
        match self {
            Self::Array(array) => Self::Array(
                array
                    .slice(index..index + 1)
                    .vortex_expect("expected output length matches the input length"),
            ),
            Self::Error => Self::Error,
        }
    }
}

/// Assert that an operation matches `expected` on batch, row, and constant inputs.
///
/// `inputs` must contain at least one array, and every input must have the same length. The
/// operation must preserve that length. Input rows must be readable independently of the operation.
///
/// Batch execution recursively materializes the result before comparison. Row execution reads a
/// fresh lazy result with [`ArrayRef::execute_scalar`]. Constant execution builds another operation
/// for each row, with all inputs replaced by one-row [`ConstantArray`]s, then materializes it.
/// [`ExpectedOutcome::Error`] requires every observed row to fail, even when batch execution fails.
/// For empty arrays, only the batch path is checked because no row or constant input exists.
///
/// # Panics
///
/// Panics if the input requirements do not hold or an execution path differs from `expected`.
#[track_caller]
pub fn assert_execution_paths(
    inputs: &[ArrayRef],
    operation: impl Fn(&[ArrayRef]) -> VortexResult<ArrayRef>,
    expected: ExpectedOutcome,
    ctx: &mut ExecutionCtx,
) {
    let first = inputs
        .first()
        .vortex_expect("execution path assertions require at least one input");
    let len = first.len();
    assert!(
        inputs.iter().all(|input| input.len() == len),
        "input arrays must have equal lengths"
    );

    if let ExpectedOutcome::Array(expected) = &expected {
        assert_eq!(
            expected.len(),
            len,
            "expected output length must match the inputs"
        );
    }

    let batch = operation(inputs).and_then(|array| execute_batch(array, ctx));
    assert_outcome(batch, &expected, "batch", ctx);

    if len == 0 {
        return;
    }

    match operation(inputs) {
        Ok(array) => {
            assert_eq!(array.len(), len, "row output length must match the inputs");
            if let ExpectedOutcome::Array(expected) = &expected {
                assert_eq!(array.dtype(), expected.dtype());
            }

            for index in 0..len {
                let row = array
                    .execute_scalar(index, ctx)
                    .map(|scalar| ConstantArray::new(scalar, 1).into_array());
                assert_outcome(row, &expected.row(index), format_args!("row {index}"), ctx);
            }
        }
        Err(error) => assert_outcome(Err(error), &expected, "row construction", ctx),
    }

    for index in 0..len {
        let constants = inputs
            .iter()
            .map(|input| {
                let scalar = input.execute_scalar(index, ctx).vortex_expect(
                    "test input rows must be readable before applying the operation",
                );
                ConstantArray::new(scalar, 1).into_array()
            })
            .collect::<Vec<_>>();
        let constant = operation(&constants).and_then(|array| execute_batch(array, ctx));
        assert_outcome(
            constant,
            &expected.row(index),
            format_args!("constant row {index}"),
            ctx,
        );
    }
}

fn execute_batch(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(array.execute::<RecursiveCanonical>(ctx)?.0.into_array())
}

#[track_caller]
fn assert_outcome(
    actual: VortexResult<ArrayRef>,
    expected: &ExpectedOutcome,
    path: impl Display,
    ctx: &mut ExecutionCtx,
) {
    match (actual, expected) {
        (Ok(actual), ExpectedOutcome::Array(expected)) => assert_arrays_eq!(actual, expected, ctx),
        (Err(_), ExpectedOutcome::Error) => {}
        (Err(error), ExpectedOutcome::Array(_)) => vortex_panic!("{path} path failed: {error}"),
        (Ok(actual), ExpectedOutcome::Error) => {
            vortex_panic!(
                "{path} path must return an error, got {} rows",
                actual.len()
            );
        }
    }
}

#[cfg(test)]
mod tests;
