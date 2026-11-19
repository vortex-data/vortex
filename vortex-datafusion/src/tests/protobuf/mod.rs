// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQL over a protobuf column, whose message fields the Vortex scan reads through the extension
//! storage.

use std::sync::Arc;

use datafusion::physical_plan::collect;
use datafusion::physical_plan::displayable;
use datafusion_common::assert_batches_eq;
use datafusion_expr::registry::FunctionRegistry;
use datafusion_functions_nested::planner::FieldAccessPlanner;
use prost::Message;
use prost_reflect::DynamicMessage;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use protox::Compiler;
use protox::file::File;
use protox::file::FileResolver;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::buffer;
use vortex::dtype::Nullability;
use vortex::dtype::extension::ExtDType;
use vortex::editions::ComponentKind;
use vortex::editions::EditionInclusion;
use vortex::editions::EditionSessionExt;
use vortex::editions::UNSTABLE;
use vortex::expr::Expression;
use vortex::expr::Lambda;
use vortex::expr::get_item;
use vortex::expr::list_transform;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::var;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::layout::VTable;
use vortex::layout::layouts::list::List;
use vortex::session::VortexSession;
use vortex_protobuf::Protobuf;
use vortex_protobuf::ProtobufDecoder;
use vortex_protobuf::ProtobufMetadata;
use vortex_protobuf::protobuf_field;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;
use crate::tests::RecordingConverter;

#[cfg(feature = "unstable_protobuf")]
mod functions;

/// A schema with a nested message and a repeated message.
const ORDER_PROTO: (&str, &str) = (
    "order.proto",
    r#"
syntax = "proto3";
package shop;

message Address {
  string city = 1;
  uint32 zip = 2;
}

message Item {
  string sku = 1;
  uint32 quantity = 2;
}

message Order {
  int64 id = 1;
  Address address = 2;
  repeated Item items = 3;
}
"#,
);

/// Resolves one import-free schema file, given by its name and source.
struct SingleFile((&'static str, &'static str));

impl FileResolver for SingleFile {
    fn open_file(&self, name: &str) -> Result<File, protox::Error> {
        let (file, source) = self.0;
        if name == file {
            File::from_source(file, source)
        } else {
            Err(protox::Error::file_not_found(name))
        }
    }
}

/// Compiles the import-free schema `file`, given by its name and source, and returns the
/// descriptor of its message `name`.
fn message_descriptor(
    file: (&'static str, &'static str),
    name: &str,
) -> anyhow::Result<MessageDescriptor> {
    let mut compiler = Compiler::with_file_resolver(SingleFile(file));
    compiler.open_file(file.0)?;
    compiler
        .descriptor_pool()
        .get_message_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("missing {name}"))
}

/// A default session that can write protobuf columns, and with them the extension layout and the
/// list layout, so the fields of the repeated `items` are written as columns of their own.
fn protobuf_session() -> anyhow::Result<VortexSession> {
    let session = VortexSession::default();
    vortex_protobuf::initialize(&session);
    // No edition includes the list layout, so the test adds it to the unstable edition.
    session.editions().declare_inclusion(EditionInclusion::new(
        ComponentKind::Layout,
        &List.id(),
        UNSTABLE,
    ))?;
    session.enable_edition(UNSTABLE)?;
    Ok(session)
}

/// Registers DataFusion's nested functions, such as `array_transform`, with `ctx`, and switches to
/// the DuckDB dialect, which parses `lambda x: ...`.
async fn enable_lambdas(ctx: &TestSessionContext) -> anyhow::Result<()> {
    datafusion_functions_nested::register_all(&mut *ctx.session.state_ref().write())?;
    ctx.session
        .sql("SET datafusion.sql_parser.dialect = 'duckdb'")
        .await?
        .collect()
        .await?;
    Ok(())
}

/// The scan expression of `array_transform(msg['items'], lambda x: x['sku'])`.
fn item_skus() -> anyhow::Result<Expression> {
    Ok(list_transform(
        protobuf_field("items", get_item("msg", root())),
        Lambda::new(["x"], get_item("sku", var("x")))?,
    ))
}

/// Asserts that every scan projection that `converter` recorded is `expected`.
fn assert_scan_projections(converter: &RecordingConverter, expected: &Expression) {
    let scan_projections = converter.scan_projections.lock();
    assert!(!scan_projections.is_empty());
    for projection in scan_projections.iter() {
        assert_eq!(projection, expected);
    }
}

/// Builds a message of type `name` from the pool of `descriptor` with the given fields.
fn build(
    descriptor: &MessageDescriptor,
    name: &str,
    fields: impl IntoIterator<Item = (&'static str, Value)>,
) -> anyhow::Result<Value> {
    let message_descriptor = descriptor
        .parent_pool()
        .get_message_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
    let mut message = DynamicMessage::new(message_descriptor);
    for (field, value) in fields {
        message.try_set_field_by_name(field, value)?;
    }
    Ok(Value::Message(message))
}

/// Builds a serialized `shop.Order` with an optional address and the SKUs of its items.
fn order(
    descriptor: &MessageDescriptor,
    id: i64,
    address: Option<(&str, u32)>,
    skus: &[&str],
) -> anyhow::Result<Vec<u8>> {
    let mut fields = vec![("id", Value::I64(id))];
    if let Some((city, zip)) = address {
        let address = build(
            descriptor,
            "shop.Address",
            [
                ("city", Value::String(city.to_owned())),
                ("zip", Value::U32(zip)),
            ],
        )?;
        fields.push(("address", address));
    }
    let items = skus
        .iter()
        .map(|sku| {
            build(
                descriptor,
                "shop.Item",
                [
                    ("sku", Value::String((*sku).to_owned())),
                    ("quantity", Value::U32(1)),
                ],
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    fields.push(("items", Value::List(items)));
    let order = build(descriptor, "shop.Order", fields)?;
    let order = order
        .as_message()
        .ok_or_else(|| anyhow::anyhow!("an order is a message"))?;
    Ok(order.encode_to_vec())
}

/// A table `{id: i64, msg: shop.Order?}` whose third message is null.
fn orders() -> anyhow::Result<ArrayRef> {
    let descriptor = message_descriptor(ORDER_PROTO, "shop.Order")?;
    let dtype = ExtDType::<Protobuf>::try_new(
        ProtobufMetadata::try_new(&descriptor)?,
        Nullability::Nullable,
    )?;

    let mut decoder = ProtobufDecoder::new(&dtype);
    decoder.append_message(&order(&descriptor, 100, Some(("Berlin", 10115)), &["a"])?)?;
    decoder.append_message(&order(&descriptor, 101, Some(("Lisbon", 1100)), &[])?)?;
    decoder.append_null()?;
    decoder.append_message(&order(&descriptor, 102, Some(("Oslo", 150)), &["b", "c"])?)?;
    decoder.append_message(&order(&descriptor, 103, None, &[])?)?;
    Ok(StructArray::from_fields(&[
        ("id", buffer![1i64, 2, 3, 4, 5].into_array()),
        ("msg", decoder.finish()?),
    ])?
    .into_array())
}

/// Writes the orders to a Vortex file with `write_session` and registers it as the table `orders`,
/// which reads with `read_session` and converts expressions with `converter`.
async fn orders_table(
    write_session: &VortexSession,
    read_session: VortexSession,
    scan_v2: bool,
    converter: Arc<dyn ExpressionConverter>,
) -> anyhow::Result<TestSessionContext> {
    protobuf_table(
        "orders",
        orders()?,
        write_session,
        read_session,
        scan_v2,
        converter,
    )
    .await
}

/// Writes `array` to a Vortex file with `write_session` and registers it as the table `name`,
/// which reads with `read_session` and converts expressions with `converter`.
async fn protobuf_table(
    name: &str,
    array: ArrayRef,
    write_session: &VortexSession,
    read_session: VortexSession,
    scan_v2: bool,
    converter: Arc<dyn ExpressionConverter>,
) -> anyhow::Result<TestSessionContext> {
    let options = VortexTableOptions {
        projection_pushdown: true,
        scan_v2,
        ..Default::default()
    };
    let factory = VortexFormatFactory::new_with_options(read_session, options)
        .with_expression_converter(converter);
    let mut ctx = TestSessionContext::new_with_factory(Arc::new(factory));
    // The workspace builds DataFusion without its nested expressions, which plan SQL field access.
    ctx.session
        .register_expr_planner(Arc::new(FieldAccessPlanner))?;

    let mut write = ObjectStoreWrite::new(
        Arc::clone(&ctx.store),
        &format!("files/{name}.vortex").into(),
    )
    .await?;
    write_session
        .write_options()
        .write(&mut write, array.to_array_stream())
        .await?;
    write.shutdown().await?;
    ctx.session
        .sql(&format!(
            "CREATE EXTERNAL TABLE {name} STORED AS vortex LOCATION '/files/'"
        ))
        .await?;
    Ok(ctx)
}

/// SQL that projects nested message fields and filters on a leaf field runs both the filter and the
/// field access in the Vortex scan.
///
/// This checks the converted expressions, not the segments that the scan reads. Both scans read
/// only the columns of the selected fields and the validity of `msg`, which the file tests of
/// `vortex-protobuf` check.
#[rstest]
#[tokio::test]
async fn test_protobuf_fields_are_read_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = orders_table(
        &session,
        session.clone(),
        scan_v2,
        Arc::clone(&converter) as _,
    )
    .await?;

    let df = ctx
        .session
        .sql(
            "SELECT id, msg['id'] AS order_id, msg['address']['city'] AS city FROM orders \
             WHERE msg['address']['zip'] > 1000 ORDER BY id",
        )
        .await?;
    let plan = df.create_physical_plan().await?;
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(!display.contains("FilterExec"), "{display}");
    let batches = collect(plan, ctx.session.task_ctx()).await?;

    assert_batches_eq!(
        [
            "+----+----------+--------+",
            "| id | order_id | city   |",
            "+----+----------+--------+",
            "| 1  | 100      | Berlin |",
            "| 2  | 101      | Lisbon |",
            "+----+----------+--------+",
        ],
        &batches
    );

    let msg = get_item("msg", root());
    let expected = pack(
        [
            ("id", get_item("id", root())),
            ("order_id", protobuf_field("id", msg.clone())),
            ("city", protobuf_field("address.city", msg)),
        ],
        Nullability::NonNullable,
    );
    assert_scan_projections(&converter, &expected);
    Ok(())
}

/// A projection that mixes message fields with an unsupported function over one of them reads only
/// those fields in the scan, once each, and DataFusion applies the function to the scan output.
///
/// The field read under `upper` follows the converter's nested field access, like the fields read
/// on their own: an absent `address`, or a null message, makes `city` and `zip` null.
#[rstest]
#[tokio::test]
async fn test_protobuf_mixed_projection_reads_needed_fields(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = orders_table(
        &session,
        session.clone(),
        scan_v2,
        Arc::clone(&converter) as _,
    )
    .await?;

    let batches = ctx
        .session
        .sql(
            "SELECT id, upper(msg['address']['city']) AS shout, msg['address']['city'] AS city, \
             msg['address']['zip'] AS zip FROM orders ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+--------+--------+-------+",
            "| id | shout  | city   | zip   |",
            "+----+--------+--------+-------+",
            "| 1  | BERLIN | Berlin | 10115 |",
            "| 2  | LISBON | Lisbon | 1100  |",
            "| 3  |        |        |       |",
            "| 4  | OSLO   | Oslo   | 150   |",
            "| 5  |        |        |       |",
            "+----+--------+--------+-------+",
        ],
        &batches
    );

    let msg = get_item("msg", root());
    let expected = pack(
        [
            ("id", get_item("id", root())),
            (
                "__vortex_expr_0",
                protobuf_field("address.city", msg.clone()),
            ),
            ("__vortex_expr_1", protobuf_field("address.zip", msg)),
        ],
        Nullability::NonNullable,
    );
    assert_scan_projections(&converter, &expected);
    Ok(())
}

/// `array_transform` reads one field of every item of the repeated `items`, through the extension
/// storage and the list layout, and the scan evaluates it whole: a null message has null items, and
/// a message without items an empty list.
#[rstest]
#[tokio::test]
async fn test_protobuf_repeated_message_field_is_read_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = orders_table(
        &session,
        session.clone(),
        scan_v2,
        Arc::clone(&converter) as _,
    )
    .await?;
    enable_lambdas(&ctx).await?;

    let batches = ctx
        .session
        .sql(
            "SELECT id, array_transform(msg['items'], lambda x: x['sku']) AS skus FROM orders \
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+--------+",
            "| id | skus   |",
            "+----+--------+",
            "| 1  | [a]    |",
            "| 2  | []     |",
            "| 3  |        |",
            "| 4  | [b, c] |",
            "| 5  | []     |",
            "+----+--------+",
        ],
        &batches
    );

    let expected = pack(
        [("id", get_item("id", root())), ("skus", item_skus()?)],
        Nullability::NonNullable,
    );
    assert_scan_projections(&converter, &expected);
    Ok(())
}

/// A lambda that the scan evaluates leaves the rest of the projection split as it is without one:
/// the scan reads the SKUs under an unsupported function, and the city beside them, rather than
/// whole messages.
#[rstest]
#[tokio::test]
async fn test_protobuf_lambda_under_unsupported_function_is_read_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter::new(session.clone()));
    let ctx = orders_table(
        &session,
        session.clone(),
        scan_v2,
        Arc::clone(&converter) as _,
    )
    .await?;
    enable_lambdas(&ctx).await?;

    let batches = ctx
        .session
        .sql(
            "SELECT id, msg['address']['city'] AS city, \
             array_has(array_transform(msg['items'], lambda x: x['sku']), 'b') AS has_b \
             FROM orders ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+--------+-------+",
            "| id | city   | has_b |",
            "+----+--------+-------+",
            "| 1  | Berlin | false |",
            "| 2  | Lisbon | false |",
            "| 3  |        |       |",
            "| 4  | Oslo   | true  |",
            "| 5  |        | false |",
            "+----+--------+-------+",
        ],
        &batches
    );

    let msg = get_item("msg", root());
    let expected = pack(
        [
            ("id", get_item("id", root())),
            ("__vortex_expr_0", protobuf_field("address.city", msg)),
            ("__vortex_expr_1", item_skus()?),
        ],
        Nullability::NonNullable,
    );
    assert_scan_projections(&converter, &expected);
    Ok(())
}

/// Whole messages read back through SQL, including a null message and an absent nested message.
#[rstest]
#[tokio::test]
async fn test_protobuf_messages_read_back(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(DefaultExpressionConverter::new(session.clone()));
    let ctx = orders_table(&session, session.clone(), scan_v2, converter).await?;

    let batches = ctx
        .session
        .sql("SELECT id, msg FROM orders ORDER BY id")
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+---------------------------------------------------------------------------------------------------+",
            "| id | msg                                                                                               |",
            "+----+---------------------------------------------------------------------------------------------------+",
            "| 1  | {id: 100, address: {city: Berlin, zip: 10115}, items: [{sku: a, quantity: 1}]}                    |",
            "| 2  | {id: 101, address: {city: Lisbon, zip: 1100}, items: []}                                          |",
            "| 3  |                                                                                                   |",
            "| 4  | {id: 102, address: {city: Oslo, zip: 150}, items: [{sku: b, quantity: 1}, {sku: c, quantity: 1}]} |",
            "| 5  | {id: 103, address: , items: []}                                                                   |",
            "+----+---------------------------------------------------------------------------------------------------+",
        ],
        &batches
    );
    Ok(())
}

/// A session that does not know the protobuf type reads protobuf columns as foreign extension
/// values, which DataFusion sees as their storage struct. The scan cannot read the storage of an
/// unknown type, so field access on it stays in DataFusion.
#[rstest]
#[tokio::test]
async fn test_protobuf_fields_read_without_protobuf_support(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let read_session = VortexSession::default();
    read_session.allow_unknown();
    let converter = Arc::new(DefaultExpressionConverter::new(read_session.clone()));
    let ctx = orders_table(&protobuf_session()?, read_session, scan_v2, converter).await?;

    let batches = ctx
        .session
        .sql(
            "SELECT id, msg['id'] AS order_id, msg['address']['city'] AS city FROM orders \
             WHERE msg['address'] IS NOT NULL ORDER BY id",
        )
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+----+----------+--------+",
            "| id | order_id | city   |",
            "+----+----------+--------+",
            "| 1  | 100      | Berlin |",
            "| 2  | 101      | Lisbon |",
            "| 4  | 102      | Oslo   |",
            "+----+----------+--------+",
        ],
        &batches
    );
    Ok(())
}
