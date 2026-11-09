// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::slice;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ExpectedOutcome;
use super::assert_execution_paths;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar_fn::fns::operators::Operator;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[track_caller]
fn assert_cast_paths(input: ArrayRef, expected: ArrayRef) {
    let dtype = expected.dtype().clone();
    assert_execution_paths(
        &[input],
        |inputs| inputs[0].cast(dtype.clone()),
        ExpectedOutcome::Array(expected),
        &mut SESSION.create_execution_ctx(),
    );
}

#[rstest]
#[case::non_nullable(buffer![1i32, 2, 3].into_array(), buffer![1i64, 2, 3].into_array())]
#[case::partially_valid(
    PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
    PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array()
)]
#[case::empty(
    PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
    PrimitiveArray::empty::<i64>(Nullability::NonNullable).into_array()
)]
fn test_cast_paths(#[case] input: ArrayRef, #[case] expected: ArrayRef) {
    assert_cast_paths(input, expected);
}

#[test]
fn test_nested_cast_paths() -> VortexResult<()> {
    let input = StructArray::from_fields(&[("value", buffer![1i32, 2, 3].into_array())])?;
    let expected = StructArray::from_fields(&[("value", buffer![1i64, 2, 3].into_array())])?;

    assert_cast_paths(input.into_array(), expected.into_array());
    Ok(())
}

#[test]
fn test_binary_paths() {
    assert_execution_paths(
        &[
            buffer![1i32, 3, 2].into_array(),
            buffer![2i32, 1, 2].into_array(),
        ],
        |inputs| inputs[0].binary(inputs[1].clone(), Operator::Lt),
        ExpectedOutcome::Array(BoolArray::from_iter([true, false, false]).into_array()),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
fn test_construction_errors() {
    assert_execution_paths(
        &[buffer![1i32, 2].into_array()],
        |_inputs| Err(vortex_err!("operation is not supported")),
        ExpectedOutcome::Error,
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
fn test_execution_errors() {
    assert_execution_paths(
        &[buffer![1000i32, 2000].into_array()],
        |inputs| inputs[0].cast(DType::Primitive(PType::I8, Nullability::NonNullable)),
        ExpectedOutcome::Error,
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
fn test_empty_inputs_only_run_batch() {
    let calls = Cell::new(0);
    let input = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();

    assert_execution_paths(
        slice::from_ref(&input),
        |inputs| {
            calls.set(calls.get() + 1);
            Ok(inputs[0].clone())
        },
        ExpectedOutcome::Array(input.clone()),
        &mut SESSION.create_execution_ctx(),
    );
    assert_eq!(calls.get(), 1);
}

#[rstest]
#[case::value(buffer![1i64, 3].into_array())]
#[case::null(PrimitiveArray::from_option_iter([Some(1i64), None]).into_array())]
#[should_panic(expected = "arrays do not match")]
fn test_rejects_incorrect_expected_values(#[case] expected: ArrayRef) {
    assert_cast_paths(buffer![1i32, 2].into_array(), expected);
}

#[test]
#[should_panic(expected = "arrays differ in type")]
fn test_rejects_incorrect_expected_dtype() {
    assert_execution_paths(
        &[buffer![1i32].into_array()],
        |inputs| Ok(inputs[0].clone()),
        ExpectedOutcome::Array(buffer![1i64].into_array()),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
#[should_panic(expected = "row construction path failed")]
fn test_rejects_mixed_success_and_error() {
    let calls = Cell::new(0);
    assert_execution_paths(
        &[buffer![1i32, 2].into_array()],
        |inputs| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 1 {
                return Err(vortex_err!("row construction failed"));
            }

            Ok(inputs[0].clone())
        },
        ExpectedOutcome::Array(buffer![1i32, 2].into_array()),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
#[should_panic(expected = "arrays do not match")]
fn test_rejects_divergent_row_values() {
    let calls = Cell::new(0);
    assert_execution_paths(
        &[buffer![1i32, 2].into_array()],
        |inputs| {
            let call = calls.get();
            calls.set(call + 1);
            if call == 1 {
                return Ok(buffer![1i32, 3].into_array());
            }

            Ok(inputs[0].clone())
        },
        ExpectedOutcome::Array(buffer![1i32, 2].into_array()),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
#[should_panic(expected = "arrays do not match")]
fn test_rejects_divergent_constant_values() {
    assert_execution_paths(
        &[buffer![1i32, 2].into_array()],
        |inputs| {
            if inputs[0].is::<Constant>() {
                return Ok(buffer![9i32].into_array());
            }

            Ok(inputs[0].clone())
        },
        ExpectedOutcome::Array(buffer![1i32, 2].into_array()),
        &mut SESSION.create_execution_ctx(),
    );
}

#[test]
#[should_panic(expected = "batch path must return an error")]
fn test_rejects_success_when_error_expected() {
    assert_execution_paths(
        &[buffer![1i32].into_array()],
        |inputs| Ok(inputs[0].clone()),
        ExpectedOutcome::Error,
        &mut SESSION.create_execution_ctx(),
    );
}
