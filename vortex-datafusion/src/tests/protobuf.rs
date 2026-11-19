// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQL over a protobuf column, whose message fields the Vortex scan reads through the extension
//! storage.

use std::sync::Arc;

use arrow_schema::Schema;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::displayable;
use datafusion_common::Result as DFResult;
use datafusion_common::assert_batches_eq;
use datafusion_expr::registry::FunctionRegistry;
use datafusion_functions_nested::planner::FieldAccessPlanner;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use parking_lot::Mutex;
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
use vortex::editions::EditionSessionExt;
use vortex::editions::UNSTABLE;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
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
use crate::convert::ProcessedProjection;

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

/// Resolves the import-free [`ORDER_PROTO`] schema.
struct OrderSchema;

impl FileResolver for OrderSchema {
    fn open_file(&self, name: &str) -> Result<File, protox::Error> {
        let (file, source) = ORDER_PROTO;
        if name == file {
            File::from_source(file, source)
        } else {
            Err(protox::Error::file_not_found(name))
        }
    }
}

/// Converts expressions with the default converter and records each scan projection.
struct RecordingConverter {
    inner: DefaultExpressionConverter,
    scan_projections: Mutex<Vec<Expression>>,
}

impl ExpressionConverter for RecordingConverter {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.inner.try_convert(expr, schema)
    }

    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let processed =
            self.inner
                .split_projection(source_projection, input_schema, output_schema)?;
        self.scan_projections
            .lock()
            .push(processed.scan_projection.clone());
        Ok(processed)
    }
}

/// A default session that can write protobuf columns, and with them the extension layout.
fn protobuf_session() -> anyhow::Result<VortexSession> {
    let session = VortexSession::default();
    vortex_protobuf::initialize(&session);
    session.enable_edition(UNSTABLE)?;
    Ok(session)
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
    let mut compiler = Compiler::with_file_resolver(OrderSchema);
    compiler.open_file(ORDER_PROTO.0)?;
    let descriptor = compiler
        .descriptor_pool()
        .get_message_by_name("shop.Order")
        .ok_or_else(|| anyhow::anyhow!("missing shop.Order"))?;
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

    let mut write =
        ObjectStoreWrite::new(Arc::clone(&ctx.store), &"files/orders.vortex".into()).await?;
    write_session
        .write_options()
        .write(&mut write, orders()?.to_array_stream())
        .await?;
    write.shutdown().await?;
    ctx.session
        .sql("CREATE EXTERNAL TABLE orders STORED AS vortex LOCATION '/files/'")
        .await?;
    Ok(ctx)
}

/// SQL that projects nested message fields and filters on a leaf field runs both the filter and the
/// field access in the Vortex scan.
///
/// This checks the converted expressions, not the segments that the scan reads. The classic scan
/// reads only the columns of the selected fields. The planner-driven scan does not push field
/// reads through a nullable struct yet, so for the nullable `msg` it reads every storage column.
#[rstest]
#[tokio::test]
async fn test_protobuf_fields_are_read_in_scan(
    #[values(false, true)] scan_v2: bool,
) -> anyhow::Result<()> {
    let session = protobuf_session()?;
    let converter = Arc::new(RecordingConverter {
        inner: DefaultExpressionConverter::new(session.clone()),
        scan_projections: Mutex::default(),
    });
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
    let scan_projections = converter.scan_projections.lock();
    assert!(!scan_projections.is_empty());
    for projection in scan_projections.iter() {
        assert_eq!(projection, &expected);
    }
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
