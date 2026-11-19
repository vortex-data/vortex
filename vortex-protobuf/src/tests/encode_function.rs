// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.protobuf.encode` scalar function.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::proto::ExprSerializeProtoExt;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SESSION;
use super::all_types_dtype;
use super::codec::sample_messages;
use super::decode_rows;
use super::encode_rows;
use crate::protobuf_encode;

/// Decodes a nullable `test.AllTypes` array whose rows are null, a message with every field set,
/// a message with no field set, and null.
fn messages() -> VortexResult<ArrayRef> {
    let dtype = all_types_dtype(Nullability::Nullable)?;
    let mut samples = sample_messages(dtype.metadata().message_descriptor())?;
    let full = samples.swap_remove(1);
    let empty = samples.swap_remove(0);
    decode_rows(&dtype, &[None, Some(full), Some(empty), None])
}

/// The function gives the bytes that `encode_messages` writes, and null for a null message.
#[test]
fn test_encode_function() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let messages = messages()?;
    let expected = VarBinViewArray::from_iter_nullable_bin(encode_rows(&messages)?);
    assert_arrays_eq!(
        messages.apply(&protobuf_encode(root()))?,
        expected,
        &mut ctx
    );
    Ok(())
}

/// A constant message encodes to its bytes in every row, and a null message to null.
#[rstest]
#[case::null(0)]
#[case::full(1)]
#[case::empty(2)]
fn test_encode_function_constant(#[case] row: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let messages = messages()?;
    let encoded = encode_rows(&messages)?.swap_remove(row);
    let constant = ConstantArray::new(messages.execute_scalar(row, &mut ctx)?, 3).into_array();
    assert_arrays_eq!(
        constant.apply(&protobuf_encode(root()))?,
        VarBinViewArray::from_iter_nullable_bin([encoded.clone(), encoded.clone(), encoded]),
        &mut ctx
    );
    Ok(())
}

/// The expression serializes, and binding it to anything but protobuf messages fails.
#[test]
fn test_encode_expression() -> VortexResult<()> {
    let expr = protobuf_encode(root());
    let deserialized = Expression::from_proto(&expr.serialize_proto()?, &SESSION)?;
    assert_eq!(deserialized, expr);

    let error = PrimitiveArray::from_iter([1i32])
        .into_array()
        .apply(&expr)
        .err()
        .ok_or_else(|| vortex_err!("i32 values were encoded"))?;
    assert!(
        error
            .to_string()
            .contains("Encoding protobuf messages requires a vortex.protobuf input, got i32"),
        "{error}"
    );
    Ok(())
}
