// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The protobuf functions in SQL, evaluated by the Vortex scan and by DataFusion.

use std::iter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::cast::AsArray;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::displayable;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::assert_batches_eq;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::ColumnarValue;
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::ScalarFunctionArgs;
use datafusion_physical_expr::PhysicalExpr;
use prost::Message;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::buffer;
use vortex::dtype::Nullability;
use vortex::dtype::extension::ExtDType;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::session::VortexSession;
use vortex_protobuf::Protobuf;
use vortex_protobuf::ProtobufDecoder;
use vortex_protobuf::ProtobufMetadata;
use vortex_protobuf::protobuf_encode;
use vortex_protobuf::protobuf_enum_name;
use vortex_protobuf::protobuf_oneof_case;

use super::build;
use super::message_descriptor;
use super::protobuf_session;
use super::protobuf_table;
use crate::common_tests::TestSessionContext;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;
use crate::protobuf::functions;
use crate::protobuf::register_functions;
use crate::tests::RecordingConverter;

/// A schema with an enum at the root and in a nested message, and a oneof in each.
const SHIPMENT_PROTO: (&str, &str) = (
    "shipment.proto",
    r#"
syntax = "proto3";
package logistics;

enum Status {
  STATUS_UNSPECIFIED = 0;
  STATUS_PLACED = 1;
  STATUS_SHIPPED = 2;
}

message Stop {
  string city = 1;
  Status status = 2;
  oneof contact {
    string phone = 3;
    string email = 4;
  }
}

message Shipment {
  Status status = 1;
  Stop destination = 2;
  oneof payment {
    string card = 3;
    uint64 voucher = 4;
  }
  string note = 5;
}
"#,
);

/// Reads enum names and oneof cases at the root and in a nested message.
const CASES_QUERY: &str = "SELECT id, protobuf_enum_name(msg, 'status') AS status, \
    protobuf_enum_name(msg, 'destination.status') AS stop_status, \
    protobuf_oneof_case(msg, 'payment') AS payment, \
    protobuf_oneof_case(msg, 'destination.contact') AS contact \
    FROM shipments ORDER BY id";

/// Filters on an enum name and a oneof case.
const FILTER_QUERY: &str = "SELECT id FROM shipments \
    WHERE protobuf_enum_name(msg, 'destination.status') = 'STATUS_SHIPPED' \
    OR protobuf_oneof_case(msg, 'payment') = 'voucher' ORDER BY id";

/// Encodes whole messages.
const ENCODE_QUERY: &str = "SELECT id, protobuf_encode(msg) AS bytes FROM shipments ORDER BY id";

/// Returns the descriptor of `logistics.Shipment`.
fn shipment_descriptor() -> anyhow::Result<MessageDescriptor> {
    message_descriptor(SHIPMENT_PROTO, "logistics.Shipment")
}

/// Returns a string value.
fn string(value: &str) -> Value {
    Value::String(value.to_owned())
}

/// The serialized shipments of the table, whose third message is null.
///
/// The first message has known enum numbers and a member of each oneof set, the second has an
/// enum number that `Status` does not define and the other oneof members, and the last has
/// neither a destination nor a payment.
fn shipment_messages() -> anyhow::Result<Vec<Option<Vec<u8>>>> {
    let descriptor = shipment_descriptor()?;
    let shipment = |fields: Vec<(&'static str, Value)>| -> anyhow::Result<Vec<u8>> {
        let shipment = build(&descriptor, "logistics.Shipment", fields)?;
        let shipment = shipment
            .as_message()
            .ok_or_else(|| anyhow::anyhow!("a shipment is a message"))?;
        Ok(shipment.encode_to_vec())
    };
    let stop = |fields: Vec<(&'static str, Value)>| build(&descriptor, "logistics.Stop", fields);
    Ok(vec![
        Some(shipment(vec![
            ("status", Value::EnumNumber(1)),
            (
                "destination",
                stop(vec![
                    ("city", string("Berlin")),
                    ("status", Value::EnumNumber(2)),
                    ("phone", string("030")),
                ])?,
            ),
            ("card", string("visa")),
        ])?),
        Some(shipment(vec![
            ("status", Value::EnumNumber(7)),
            (
                "destination",
                stop(vec![
                    ("city", string("Oslo")),
                    ("email", string("post@example.com")),
                ])?,
            ),
            ("voucher", Value::U64(9)),
        ])?),
        None,
        Some(shipment(vec![("note", string("late"))])?),
    ])
}

/// A table `{id: i64, msg: logistics.Shipment?}` of the [`shipment_messages`].
fn shipments() -> anyhow::Result<ArrayRef> {
    let dtype = ExtDType::<Protobuf>::try_new(
        ProtobufMetadata::try_new(&shipment_descriptor()?)?,
        Nullability::Nullable,
    )?;
    let mut decoder = ProtobufDecoder::new(&dtype);
    for message in shipment_messages()? {
        match message {
            Some(message) => decoder.append_message(&message)?,
            None => decoder.append_null()?,
        }
    }
    Ok(StructArray::from_fields(&[
        ("id", buffer![1i64, 2, 3, 4].into_array()),
        ("msg", decoder.finish()?),
    ])?
    .into_array())
}

/// Registers the shipments as the table `shipments`, which reads with `read_session` and converts
/// expressions with `converter`, and registers the protobuf functions.
async fn shipments_table(
    read_session: VortexSession,
    scan_v2: bool,
    converter: Arc<dyn ExpressionConverter>,
) -> anyhow::Result<TestSessionContext> {
    let mut ctx = protobuf_table(
        "shipments",
        shipments()?,
        &protobuf_session()?,
        read_session,
        scan_v2,
        converter,
    )
    .await?;
    register_functions(&mut ctx.session)?;
    Ok(ctx)
}

/// Plans and runs `query`, returning its results and the indented physical plan.
async fn run(ctx: &TestSessionContext, query: &str) -> anyhow::Result<(Vec<RecordBatch>, String)> {
    let plan = ctx.session.sql(query).await?.create_physical_plan().await?;
    let display = displayable(plan.as_ref()).indent(true).to_string();
    Ok((collect(plan, ctx.session.task_ctx()).await?, display))
}

/// Converts no expression, so DataFusion evaluates every expression over the raw columns.
struct NoConversion;

impl ExpressionConverter for NoConversion {
    fn try_convert(
        &self,
        _expr: &Arc<dyn PhysicalExpr>,
        _schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        Ok(None)
    }
}

/// Why DataFusion, rather than the scan, evaluates the protobuf functions.
#[derive(Clone, Copy, Debug)]
enum Evaluation {
    /// The converter converts no expression.
    Unconverted,
    /// The read session does not know the protobuf type, so the scan reads foreign extension
    /// values.
    Foreign,
}

/// Enum names and oneof cases, at the root and in a nested message, are evaluated in the scan,
/// which reads only the fields that each call uses.
#[rstest]
#[tokio::test]
async fn test_protobuf_enum_names_and_oneof_cases_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = shipments_table(session, scan_v2, Arc::clone(&converter) as _).await?;

    let (batches, _) = run(&ctx, CASES_QUERY).await?;
    assert_batches_eq!(
        [
            "+----+--------------------+--------------------+---------+---------+",
            "| id | status             | stop_status        | payment | contact |",
            "+----+--------------------+--------------------+---------+---------+",
            "| 1  | STATUS_PLACED      | STATUS_SHIPPED     | card    | phone   |",
            "| 2  |                    | STATUS_UNSPECIFIED | voucher | email   |",
            "| 3  |                    |                    |         |         |",
            "| 4  | STATUS_UNSPECIFIED |                    |         |         |",
            "+----+--------------------+--------------------+---------+---------+",
        ],
        &batches
    );

    let metadata = ProtobufMetadata::try_new(&shipment_descriptor()?)?;
    let msg = get_item("msg", root());
    let expected = pack(
        [
            ("id", get_item("id", root())),
            (
                "status",
                protobuf_enum_name(&metadata, "status", msg.clone())?,
            ),
            (
                "stop_status",
                protobuf_enum_name(&metadata, "destination.status", msg.clone())?,
            ),
            (
                "payment",
                protobuf_oneof_case(&metadata, "payment", msg.clone())?,
            ),
            (
                "contact",
                protobuf_oneof_case(&metadata, "destination.contact", msg)?,
            ),
        ],
        Nullability::NonNullable,
    );
    let scan_projections = converter.scan_projections.lock();
    assert!(!scan_projections.is_empty());
    for projection in scan_projections.iter() {
        assert_eq!(projection, &expected);
    }
    Ok(())
}

/// A filter on an enum name and a oneof case runs in the scan.
#[rstest]
#[tokio::test]
async fn test_protobuf_function_filter_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(DefaultExpressionConverter::new(session.clone()));
    let ctx = shipments_table(session, scan_v2, converter).await?;

    let (batches, display) = run(&ctx, FILTER_QUERY).await?;
    assert!(!display.contains("FilterExec"), "{display}");
    assert_batches_eq!(
        ["+----+", "| id |", "+----+", "| 1  |", "| 2  |", "+----+"],
        &batches
    );
    Ok(())
}

/// Encoding in the scan gives the wire bytes of each message, and null for a null message.
#[rstest]
#[tokio::test]
async fn test_protobuf_encode_in_scan(#[values(false, true)] scan_v2: bool) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = shipments_table(session, scan_v2, Arc::clone(&converter) as _).await?;

    let (batches, _) = run(&ctx, ENCODE_QUERY).await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let bytes = batch
        .column_by_name("bytes")
        .ok_or_else(|| anyhow::anyhow!("missing bytes"))?
        .as_binary::<i32>()
        .iter()
        .map(|bytes| bytes.map(<[u8]>::to_vec))
        .collect::<Vec<_>>();
    assert_eq!(bytes, shipment_messages()?);

    let expected = pack(
        [
            ("id", get_item("id", root())),
            ("bytes", protobuf_encode(get_item("msg", root()))),
        ],
        Nullability::NonNullable,
    );
    let scan_projections = converter.scan_projections.lock();
    assert!(!scan_projections.is_empty());
    for projection in scan_projections.iter() {
        assert_eq!(projection, &expected);
    }
    Ok(())
}

/// DataFusion evaluates the functions to the same results as the scan.
#[rstest]
#[tokio::test]
async fn test_protobuf_functions_in_datafusion(
    #[values(false, true)] scan_v2: bool,
    #[values(Evaluation::Unconverted, Evaluation::Foreign)] evaluation: Evaluation,
    #[values(CASES_QUERY, FILTER_QUERY, ENCODE_QUERY)] query: &str,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(DefaultExpressionConverter::new(session.clone()));
    let scanned = shipments_table(session.clone(), scan_v2, converter).await?;
    let (expected, _) = run(&scanned, query).await?;

    let (batches, display) = match evaluation {
        Evaluation::Unconverted => {
            let ctx = shipments_table(session, scan_v2, Arc::new(NoConversion)).await?;
            run(&ctx, query).await?
        }
        Evaluation::Foreign => {
            let foreign = VortexSession::default();
            foreign.allow_unknown();
            let converter = Arc::new(RecordingConverter::new(foreign.clone()));
            let ctx = shipments_table(foreign, scan_v2, Arc::clone(&converter) as _).await?;
            let result = run(&ctx, query).await?;
            for projection in converter.scan_projections.lock().iter() {
                assert!(
                    !projection.to_string().contains("vortex.protobuf"),
                    "{projection}"
                );
            }
            result
        }
    };
    assert_eq!(
        display.contains("FilterExec"),
        query.contains("WHERE"),
        "{display}"
    );
    assert_eq!(
        pretty_format_batches(&batches)?.to_string(),
        pretty_format_batches(&expected)?.to_string()
    );
    Ok(())
}

/// A call over one message, as DataFusion makes for a scalar argument, gives the row that the
/// call over an array of messages gives.
#[rstest]
#[case::enum_name("protobuf_enum_name", Some("destination.status"))]
#[case::oneof_case("protobuf_oneof_case", Some("payment"))]
#[case::encode("protobuf_encode", None)]
#[tokio::test]
async fn test_protobuf_function_of_scalar_message(
    #[case] name: &str,
    #[case] path: Option<&str>,
) -> anyhow::Result<()> {
    let ctx = shipments_table(protobuf_session()?, false, Arc::new(NoConversion)).await?;
    let (batches, _) = run(&ctx, "SELECT msg FROM shipments ORDER BY id").await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;
    let messages = Arc::clone(batch.column(0));
    let function = functions()
        .into_iter()
        .find(|function| function.name() == name)
        .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;

    let path = path.map(|path| ScalarValue::Utf8(Some(path.to_owned())));
    let arg_fields = iter::once(Arc::new(batch.schema().field(0).clone()))
        .chain(
            path.as_ref()
                .map(|_| Arc::new(Field::new("path", DataType::Utf8, false))),
        )
        .collect::<Vec<_>>();
    let return_field = function.return_field_from_args(ReturnFieldArgs {
        arg_fields: &arg_fields,
        scalar_arguments: &[None, path.as_ref()][..arg_fields.len()],
    })?;
    let invoke = |messages: ColumnarValue, number_rows: usize| {
        function.invoke_with_args(ScalarFunctionArgs {
            args: iter::once(messages)
                .chain(path.clone().map(ColumnarValue::Scalar))
                .collect(),
            arg_fields: arg_fields.clone(),
            number_rows,
            return_field: Arc::clone(&return_field),
            config_options: Arc::new(ConfigOptions::new()),
        })
    };

    let expected = invoke(ColumnarValue::Array(Arc::clone(&messages)), messages.len())?
        .into_array(messages.len())?;
    for row in 0..messages.len() {
        let message = ScalarValue::try_from_array(&messages, row)?;
        let ColumnarValue::Scalar(result) = invoke(ColumnarValue::Scalar(message), 1)? else {
            anyhow::bail!("{name} of a scalar message returned an array");
        };
        assert_eq!(
            result,
            ScalarValue::try_from_array(&expected, row)?,
            "row {row}"
        );
    }
    Ok(())
}

/// Planning fails for a call whose message argument is not a protobuf column, or whose path is
/// not a string literal that names a field of the right kind.
#[rstest]
#[case::not_an_enum(
    "protobuf_enum_name(msg, 'destination')",
    "protobuf_enum_name: Protobuf field destination is not a singular enum field"
)]
#[case::missing_field(
    "protobuf_enum_name(msg, 'destination.missing')",
    "protobuf_enum_name: Protobuf message logistics.Stop has no field missing"
)]
#[case::not_a_oneof(
    "protobuf_oneof_case(msg, 'status')",
    "protobuf_oneof_case: Protobuf message logistics.Shipment has no oneof status"
)]
#[case::column_path(
    "protobuf_oneof_case(msg, CAST(id AS VARCHAR))",
    "protobuf_oneof_case requires a string literal path"
)]
#[case::null_path(
    "protobuf_enum_name(msg, NULL)",
    "protobuf_enum_name requires a string path"
)]
#[case::not_messages(
    "protobuf_encode(id)",
    "protobuf_encode requires vortex.protobuf messages, got Int64"
)]
#[case::nested_message(
    "protobuf_enum_name(msg['destination'], 'status')",
    "protobuf_enum_name requires vortex.protobuf messages, got Struct"
)]
#[tokio::test]
async fn test_protobuf_function_rejects_arguments(
    #[case] call: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(DefaultExpressionConverter::new(session.clone()));
    let ctx = shipments_table(session, false, converter).await?;

    let error = ctx
        .session
        .sql(&format!("SELECT {call} FROM shipments"))
        .await
        .err()
        .ok_or_else(|| anyhow::anyhow!("{call} was planned"))?;
    assert!(error.to_string().contains(expected), "{error}");
    Ok(())
}
