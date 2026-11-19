// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The names of enum values and the set members of oneofs.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::lit;
use vortex_array::expr::proto::ExprSerializeProtoExt;
use vortex_array::expr::root;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::CLOSED_PROTO;
use super::SESSION;
use super::TEST_PROTO;
use super::UNRELATED_PROTO;
use super::codec::sample_messages;
use super::decode_rows;
use super::len_field;
use super::message;
use super::protobuf_dtype_with;
use super::varint_field;
use crate::EnumName;
use crate::EnumNameOptions;
use crate::MappingOptions;
use crate::ProtobufMetadata;
use crate::protobuf_enum_name;
use crate::protobuf_oneof_case;

/// A schema with an enum that has aliases, enum fields of several kinds and a oneof in a nested
/// message.
const JOB_PROTO: (&str, &str) = (
    "job.proto",
    r#"
syntax = "proto3";
package job;

enum Status {
  option allow_alias = true;
  STATUS_UNKNOWN = 0;
  STATUS_STARTED = 1;
  STATUS_RUNNING = 1;
  STATUS_DONE = 2;
}

message Step {
  Status status = 1;
  oneof target {
    string host = 2;
    int32 port = 3;
  }
}

message Job {
  Status status = 1;
  optional Status previous = 2;
  Step step = 3;
  repeated Status history = 4;
  Job parent = 5;
}
"#,
);

/// The mapping options of the tests, which accept `job.Job`, a message that contains itself.
fn options() -> MappingOptions {
    MappingOptions {
        max_recursion_depth: Some(0),
        ..MappingOptions::default()
    }
}

/// Decodes `rows` of the message `name` compiled from `sources`, and applies the expression that
/// `build` makes from the dtype metadata.
fn evaluate(
    sources: &[(&str, &str)],
    name: &str,
    rows: Vec<Vec<u8>>,
    build: impl FnOnce(&ProtobufMetadata) -> VortexResult<Expression>,
) -> VortexResult<ArrayRef> {
    let dtype = protobuf_dtype_with(sources, name, Nullability::NonNullable, options())?;
    let rows = rows.into_iter().map(Some).collect::<Vec<_>>();
    let expr = build(dtype.metadata())?;
    decode_rows(&dtype, &rows)?.apply(&expr)
}

/// The test messages of `test.AllTypes`, whose `f_enum` numbers are 0, 1, 0 and 7, and whose
/// `choice` oneof is unset, `c_inner`, `c_text` and `c_text`.
fn all_types_rows() -> VortexResult<Vec<Vec<u8>>> {
    sample_messages(&message(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes")?)
}

/// The names of the values of an enum field, null for a null field and for a number that the
/// enum does not define. An aliased number has the name declared first.
#[rstest]
#[case::open_enum(
    &[TEST_PROTO, UNRELATED_PROTO],
    "test.AllTypes",
    all_types_rows(),
    "f_enum",
    [Some("COLOR_UNSPECIFIED"), Some("COLOR_RED"), Some("COLOR_UNSPECIFIED"), None]
)]
#[case::aliases(
    &[JOB_PROTO],
    "job.Job",
    Ok(vec![Vec::new(), varint_field(1, 1), varint_field(1, 2), varint_field(1, 9)]),
    "status",
    [Some("STATUS_UNKNOWN"), Some("STATUS_STARTED"), Some("STATUS_DONE"), None]
)]
#[case::optional_field(
    &[JOB_PROTO],
    "job.Job",
    Ok(vec![Vec::new(), varint_field(2, 0), varint_field(2, 2), varint_field(2, 3)]),
    "previous",
    [None, Some("STATUS_UNKNOWN"), Some("STATUS_DONE"), None]
)]
#[case::nested_field(
    &[JOB_PROTO],
    "job.Job",
    Ok(vec![Vec::new(), len_field(3, &[]), len_field(3, &varint_field(1, 2)), Vec::new()]),
    "step.status",
    [None, Some("STATUS_UNKNOWN"), Some("STATUS_DONE"), None]
)]
#[case::closed_enum(
    &[CLOSED_PROTO],
    "closed.Closed",
    Ok(vec![Vec::new(), varint_field(1, 5), varint_field(1, 4), varint_field(1, 1)]),
    "mode",
    [None, Some("MODE_C"), None, Some("MODE_A")]
)]
fn test_enum_names(
    #[case] sources: &[(&str, &str)],
    #[case] name: &str,
    #[case] rows: VortexResult<Vec<Vec<u8>>>,
    #[case] path: &str,
    #[case] expected: [Option<&str>; 4],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let names = evaluate(sources, name, rows?, |metadata| {
        protobuf_enum_name(metadata, path, root())
    })?;
    assert_arrays_eq!(
        names,
        VarBinViewArray::from_iter_nullable_str(expected),
        &mut ctx
    );
    Ok(())
}

/// The enum name of a path that does not end in a singular enum field fails.
#[rstest]
#[case::not_an_enum("step", "Protobuf field step is not a singular enum field")]
#[case::repeated_enum("history", "Protobuf field history is not a singular enum field")]
#[case::missing_field("step.missing", "Protobuf message job.Step has no field missing")]
#[case::through_scalar(
    "status.missing",
    "Protobuf field status of job.Job is not a singular message stored as a struct"
)]
#[case::through_cut(
    "parent.status",
    "Protobuf field parent of job.Job is not a singular message stored as a struct"
)]
fn test_enum_name_rejects_paths(#[case] path: &str, #[case] expected: &str) -> VortexResult<()> {
    let dtype = protobuf_dtype_with(&[JOB_PROTO], "job.Job", Nullability::NonNullable, options())?;
    let error = protobuf_enum_name(dtype.metadata(), path, root())
        .err()
        .ok_or_else(|| vortex_err!("{path} was accepted"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}

/// The enum name expression serializes with its enum values, and a constant number maps to a
/// constant name.
#[test]
fn test_enum_name_expression() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let enum_type = message(&[JOB_PROTO], "job.Job")?
        .parent_pool()
        .get_enum_by_name("job.Status")
        .ok_or_else(|| vortex_err!("missing job.Status"))?;
    let expr = EnumName.new_expr(EnumNameOptions::new(&enum_type), [root()]);
    assert_eq!(
        expr.to_string(),
        "vortex.protobuf.enum_name($, opts=job.Status)"
    );
    let deserialized = Expression::from_proto(&expr.serialize_proto()?, &SESSION)?;
    assert_eq!(deserialized, expr);

    let numbers = PrimitiveArray::from_option_iter([Some(1i32), None, Some(-1)]).into_array();
    assert_arrays_eq!(
        numbers.clone().apply(&deserialized)?,
        VarBinViewArray::from_iter_nullable_str([Some("STATUS_STARTED"), None, None]),
        &mut ctx
    );
    let constant = EnumName.new_expr(EnumNameOptions::new(&enum_type), [lit(2i32)]);
    assert_arrays_eq!(
        numbers.apply(&constant)?,
        VarBinViewArray::from_iter_nullable_str([Some("STATUS_DONE"); 3]),
        &mut ctx
    );

    let error = PrimitiveArray::from_iter([1i64])
        .into_array()
        .apply(&expr)
        .err()
        .ok_or_else(|| vortex_err!("i64 numbers were accepted"))?;
    assert!(
        error
            .to_string()
            .contains("The names of protobuf enum job.Status need i32 numbers, got i64"),
        "{error}"
    );
    Ok(())
}

/// The name of the set member of a oneof, null where no member is set or a message on the path
/// is null.
#[rstest]
#[case::oneof(
    &[TEST_PROTO, UNRELATED_PROTO],
    "test.AllTypes",
    all_types_rows(),
    "choice",
    [None, Some("c_inner"), Some("c_text"), Some("c_text")]
)]
#[case::nested_oneof(
    &[JOB_PROTO],
    "job.Job",
    Ok(vec![
        Vec::new(),
        len_field(3, &[]),
        len_field(3, &len_field(2, b"")),
        len_field(3, &[len_field(2, b"h"), varint_field(3, 0)].concat()),
    ]),
    "step.target",
    [None, None, Some("host"), Some("port")]
)]
#[case::synthetic_oneof(
    &[JOB_PROTO],
    "job.Job",
    Ok(vec![Vec::new(), varint_field(2, 0), Vec::new(), varint_field(2, 1)]),
    "_previous",
    [None, Some("previous"), None, Some("previous")]
)]
fn test_oneof_case(
    #[case] sources: &[(&str, &str)],
    #[case] name: &str,
    #[case] rows: VortexResult<Vec<Vec<u8>>>,
    #[case] path: &str,
    #[case] expected: [Option<&str>; 4],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let cases = evaluate(sources, name, rows?, |metadata| {
        protobuf_oneof_case(metadata, path, root())
    })?;
    assert_arrays_eq!(
        cases,
        VarBinViewArray::from_iter_nullable_str(expected),
        &mut ctx
    );
    Ok(())
}

/// The oneof case of a path that does not end in a oneof fails.
#[rstest]
#[case::missing_oneof("step.missing", "Protobuf message job.Step has no oneof missing")]
#[case::field_not_oneof("status", "Protobuf message job.Job has no oneof status")]
#[case::through_list(
    "history.target",
    "Protobuf field history of job.Job is not a singular message stored as a struct"
)]
fn test_oneof_case_rejects_paths(#[case] path: &str, #[case] expected: &str) -> VortexResult<()> {
    let dtype = protobuf_dtype_with(&[JOB_PROTO], "job.Job", Nullability::NonNullable, options())?;
    let error = protobuf_oneof_case(dtype.metadata(), path, root())
        .err()
        .ok_or_else(|| vortex_err!("{path} was accepted"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}
