// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Protobuf columns in Vortex files written with the default write strategy.

use std::fmt::Write as _;
use std::sync::Arc;

use futures::TryStreamExt;
use prost::Message;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::fill_null;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::list_transform;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::var;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_edition::UNSTABLE;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_file::v2::ScanDriver;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutRef;
use vortex_layout::VTable;
use vortex_layout::layouts::extension::Extension;
use vortex_layout::layouts::list::List;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::scan::metrics::ScanUsage;
use vortex_layout::scan::metrics::ScanUsageSnapshot;
use vortex_layout::scan::v2::ScanBuilder as PlannedScanBuilder;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::TEST_PROTO;
use super::UNRELATED_PROTO;
use super::build;
use super::codec::sample_messages;
use super::decode_rows;
use super::message;
use super::protobuf_dtype_with;
use super::varint_field;
use super::well_known::KNOWN_PROTO;
use super::well_known::dedicated;
use super::well_known::known_messages;
use crate::MappingOptions;
use crate::Protobuf;
use crate::ProtobufDecoder;
use crate::ProtobufMetadata;
use crate::protobuf_field;

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

/// The rows of each block that the write strategy compresses and writes as one chunk.
const ROW_BLOCK_SIZE: usize = 2048;

/// Enough orders for several blocks.
const ORDERS: usize = 10_000;

const CITIES: [&str; 3] = ["Berlin", "Lisbon", "Oslo"];

/// The two scan executors of a Vortex file.
#[derive(Clone, Copy, Debug)]
enum Executor {
    /// The layout reader scan of [`VortexFile::scan`].
    Reader,
    /// The planner-driven scan of [`VortexFile::scan_file`].
    Planner,
}

/// A session that writes protobuf columns with the default write strategy and the unstable
/// edition, which then uses the extension layout. With `list_layout`, it also uses the list layout,
/// so a repeated message field is written as one column per field of the message.
fn file_session(handle: Handle, list_layout: bool) -> VortexResult<VortexSession> {
    let session = VortexSession::empty().with_handle(handle);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    vortex_file::register_default_encodings(&session);
    crate::initialize(&session);
    if list_layout {
        // No edition includes the list layout, so the test adds it to the unstable edition.
        session.editions().declare_inclusion(EditionInclusion::new(
            ComponentKind::Layout,
            &List.id(),
            UNSTABLE,
        ))?;
    }
    session.enable_edition(CORE_2026_08_3)?;
    session.enable_edition(UNSTABLE)?;
    Ok(session)
}

/// Serialized order `i`: it has `i % 3` items, and no address when `i % 4 == 3`.
fn order(descriptor: &MessageDescriptor, i: usize) -> VortexResult<Vec<u8>> {
    let id = i64::try_from(i)?;
    let mut fields = vec![("id", Value::I64(id))];
    if i % 4 != 3 {
        let address = build(
            descriptor,
            "shop.Address",
            [
                ("city", Value::String(CITIES[i % CITIES.len()].to_owned())),
                ("zip", Value::U32(u32::try_from(i)?)),
            ],
        )?;
        fields.push(("address", Value::Message(address)));
    }
    let items = (0..i % 3)
        .map(|item| {
            build(
                descriptor,
                "shop.Item",
                [
                    ("sku", Value::String(format!("sku-{i}-{item}"))),
                    ("quantity", Value::U32(u32::try_from(item + 1)?)),
                ],
            )
            .map(Value::Message)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    fields.push(("items", Value::List(items)));
    Ok(build(descriptor, "shop.Order", fields)?.encode_to_vec())
}

/// Decodes `messages` into a protobuf column of the message `descriptor`. With a nullable
/// `nullability`, every seventh message is null.
fn messages_column(
    descriptor: MessageDescriptor,
    messages: &[Vec<u8>],
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let dtype =
        ExtDType::<Protobuf>::try_new(ProtobufMetadata::try_new(&descriptor)?, nullability)?;
    let mut decoder = ProtobufDecoder::with_capacity(&dtype, messages.len());
    for (row, message) in messages.iter().enumerate() {
        if nullability.is_nullable() && row % 7 == 0 {
            decoder.append_null()?;
        } else {
            decoder.append_message(message)?;
        }
    }
    decoder.finish()
}

/// A table `{id: i64, msg: shop.Order}` of [`ORDERS`] orders.
fn orders_table(nullability: Nullability) -> VortexResult<ArrayRef> {
    let descriptor = message(&[ORDER_PROTO], "shop.Order")?;
    let orders = (0..ORDERS)
        .map(|i| order(&descriptor, i))
        .collect::<VortexResult<Vec<_>>>()?;
    let ids: Buffer<i64> = (0..).take(ORDERS).collect();
    Ok(StructArray::from_fields(&[
        ("id", ids.into_array()),
        ("msg", messages_column(descriptor, &orders, nullability)?),
    ])?
    .into_array())
}

/// Writes `table` to an in-memory Vortex file with the session's default write strategy, in
/// blocks of [`ROW_BLOCK_SIZE`] rows.
async fn write(session: &VortexSession, table: &ArrayRef) -> VortexResult<VortexFile> {
    let strategy = WriteStrategyBuilder::from_session(session)
        .with_row_block_size(ROW_BLOCK_SIZE)
        .with_data_block_target_bytes(None)
        .build();
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut buffer, table.to_array_stream())
        .await?;
    session.open_options().open_buffer(buffer)
}

/// Scans `file` with `executor`, keeping the rows that match `filter` and evaluating `projection`.
async fn scan(
    file: &VortexFile,
    executor: Executor,
    handle: Handle,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<ArrayRef> {
    let projection = projection.bind(file.dtype())?;
    let filter = filter.map(|filter| filter.bind(file.dtype())).transpose()?;
    match executor {
        Executor::Reader => {
            file.scan()?
                .with_projection(projection)
                .with_some_filter(filter)
                .into_array_stream()?
                .read_all()
                .await
        }
        Executor::Planner => {
            let dtype = projection.dtype().clone();
            let builder = PlannedScanBuilder::new(file.session().clone(), file.scan_file())
                .with_projection(projection)
                .with_some_filter(filter);
            let chunks: Vec<ArrayRef> = ScanDriver::new(handle, 1, true)
                .run(builder)
                .try_collect()
                .await?;
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
    }
}

/// Scans `file` as [`scan`] does, and counts the segments the scan reads.
async fn scan_counting(
    file: &VortexFile,
    executor: Executor,
    handle: Handle,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<(ArrayRef, ScanUsageSnapshot)> {
    let usage = Arc::new(ScanUsage::default());
    let file = file.clone().with_usage(Arc::clone(&usage));
    let array = scan(&file, executor, handle, projection, filter).await?;
    Ok((array, usage.snapshot()))
}

/// The `field` of every item of each order: `list_transform(items, x -> x.field)`, which is how
/// DataFusion's `array_transform` reaches Vortex.
fn item_field(field: &str) -> VortexResult<Expression> {
    Ok(list_transform(
        protobuf_field("items", get_item("msg", root())),
        Lambda::new(["x"], get_item(field, var("x")))?,
    ))
}

/// Whether `layout` splits a value into columns: a struct, list or extension layout.
fn is_structural(layout: &LayoutRef) -> bool {
    let id = layout.encoding_id();
    id == Struct.id() || id == List.id() || id == Extension.id()
}

/// Renders the columns that the struct, list and extension layouts under `layout` split a value
/// into, one line per child. The children of any other layout belong to a single column and are
/// not shown.
fn columns(layout: &LayoutRef) -> VortexResult<String> {
    let mut out = String::new();
    write_columns(layout, 0, &mut out)?;
    Ok(out)
}

/// Writes the children of `layout` for [`columns`], indented by `depth` levels.
fn write_columns(layout: &LayoutRef, depth: usize, out: &mut String) -> VortexResult<()> {
    if !is_structural(layout) {
        return Ok(());
    }
    for (name, child) in layout.child_names().zip(layout.children()?) {
        let kind = if is_structural(&child) {
            child.encoding_id().to_string()
        } else {
            "column".to_owned()
        };
        writeln!(out, "{:indent$}{name}: {kind}", "", indent = 2 * depth)
            .map_err(|err| vortex_err!("{err}"))?;
        write_columns(&child, depth + 1, out)?;
    }
    Ok(())
}

/// The writer splits a protobuf column into one column per message field, including the fields of
/// nested messages and, with the list layout, of the messages in a repeated field.
#[test]
fn test_file_splits_message_fields() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle, true)?;
        let file = write(&session, &orders_table(Nullability::Nullable)?).await?;
        insta::assert_snapshot!(columns(file.footer().layout())?, @r"
        id: column
        msg: vortex.ext
          storage: vortex.struct
            validity: column
            id: column
            address: vortex.struct
              validity: column
              city: column
              zip: column
            items: vortex.list
              elements: vortex.struct
                sku: column
                quantity: column
              offsets: column
        ");
        Ok(())
    })
}

/// Every protobuf value reads back from a file, with and without the list layout.
#[rstest]
fn test_file_round_trip(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
    #[values(false, true)] list_layout: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), list_layout)?;
        let descriptor = message(&[TEST_PROTO, UNRELATED_PROTO], "test.AllTypes")?;
        let samples = sample_messages(&descriptor)?;
        // Repeat the samples so that every sample is also at a valid row when some are null.
        let messages = samples
            .iter()
            .cycle()
            .take(4 * samples.len())
            .cloned()
            .collect::<Vec<_>>();
        let msg = messages_column(descriptor, &messages, nullability)?;
        let table = StructArray::from_fields(&[("msg", msg)])?.into_array();
        let file = write(&session, &table).await?;

        let actual = scan(&file, executor, handle, &root(), None).await?;
        assert_arrays_eq!(actual, table, &mut session.create_execution_ctx());
        Ok(())
    })
}

/// Reading one message field through the field helper matches evaluating it in memory.
#[rstest]
fn test_file_reads_field(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
    #[values(false, true)] list_layout: bool,
    #[values("id", "address", "address.city", "items")] path: &str,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), list_layout)?;
        let table = orders_table(Nullability::Nullable)?;
        let file = write(&session, &table).await?;

        let field = protobuf_field(path, get_item("msg", root()));
        let actual = scan(&file, executor, handle, &field, None).await?;
        assert_arrays_eq!(
            actual,
            table.apply(&field)?,
            &mut session.create_execution_ctx()
        );
        Ok(())
    })
}

/// A filter on a leaf message field keeps the rows that the same filter keeps in memory.
#[rstest]
fn test_file_filters_on_leaf_field(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
    #[values(false, true)] list_layout: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), list_layout)?;
        let table = orders_table(Nullability::Nullable)?;
        let file = write(&session, &table).await?;

        let msg = get_item("msg", root());
        let filter = gt(protobuf_field("address.zip", msg.clone()), lit(9_000u32));
        let projection = pack(
            [
                ("id", get_item("id", root())),
                ("city", protobuf_field("address.city", msg)),
            ],
            Nullability::NonNullable,
        );
        let actual = scan(&file, executor, handle, &projection, Some(&filter)).await?;

        let mut ctx = session.create_execution_ctx();
        let keep = table
            .clone()
            .apply(&fill_null(filter, lit(false)))?
            .execute::<Mask>(&mut ctx)?;
        assert!(keep.true_count() > 0 && keep.true_count() < ORDERS);
        let expected = table.filter(keep)?.apply(&projection)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    })
}

/// Protobuf columns with unknown fields and the dedicated forms of well-known types, which hold
/// nested timestamp and JSON extension columns, read back from a file, and so do their fields.
#[rstest]
fn test_file_round_trip_with_mapping_options(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), true)?;
        let options = MappingOptions {
            preserve_unknown_fields: true,
            ..dedicated()
        };
        let dtype = protobuf_dtype_with(
            &[KNOWN_PROTO],
            "known.Known",
            Nullability::Nullable,
            options,
        )?;
        let samples = known_messages(dtype.metadata().message_descriptor())?;
        let rows = (0..4 * samples.len())
            .map(|row| {
                (row % 5 != 4).then(|| {
                    [
                        samples[row % samples.len()].as_slice(),
                        &varint_field(99, 7),
                    ]
                    .concat()
                })
            })
            .collect::<Vec<_>>();
        let table = StructArray::from_fields(&[("msg", decode_rows(&dtype, &rows)?)])?.into_array();
        let file = write(&session, &table).await?;

        let mut ctx = session.create_execution_ctx();
        let actual = scan(&file, executor, handle.clone(), &root(), None).await?;
        assert_arrays_eq!(actual, table, &mut ctx);
        for path in ["created", "attributes", "int32_value", "$unknown"] {
            let field = protobuf_field(path, get_item("msg", root()));
            let actual = scan(&file, executor, handle.clone(), &field, None).await?;
            assert_arrays_eq!(actual, table.clone().apply(&field)?, &mut ctx);
        }
        Ok(())
    })
}

/// Reading one field of every item of a repeated message field matches evaluating it in memory,
/// and reads that field's column, not the other item columns.
///
/// The list layout reads the offsets for the lengths, so reading the SKUs and then the quantities
/// reads exactly what reading the whole items and then the lengths reads.
#[rstest]
fn test_file_reads_repeated_message_field(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), true)?;
        let table = orders_table(nullability)?;
        let file = write(&session, &table).await?;
        let mut ctx = session.create_execution_ctx();

        let skus = item_field("sku")?;
        let (actual, skus_read) =
            scan_counting(&file, executor, handle.clone(), &skus, None).await?;
        assert_arrays_eq!(actual, table.clone().apply(&skus)?, &mut ctx);

        let quantities = item_field("quantity")?;
        let (actual, quantities_read) =
            scan_counting(&file, executor, handle.clone(), &quantities, None).await?;
        assert_arrays_eq!(actual, table.clone().apply(&quantities)?, &mut ctx);

        let items = protobuf_field("items", get_item("msg", root()));
        let (_, items_read) = scan_counting(&file, executor, handle.clone(), &items, None).await?;
        let lengths = list_length(items);
        let (_, lengths_read) = scan_counting(&file, executor, handle, &lengths, None).await?;
        assert!(skus_read.segments_requested < items_read.segments_requested);
        assert_eq!(
            skus_read.segments_requested + quantities_read.segments_requested,
            items_read.segments_requested + lengths_read.segments_requested
        );
        Ok(())
    })
}

/// A filter on the SKUs of the items keeps the rows that the same filter keeps in memory, and reads
/// no more than projecting the SKUs and ids does, so it does not read the item quantities.
#[rstest]
fn test_file_filters_on_repeated_message_field(
    #[values(Executor::Reader, Executor::Planner)] executor: Executor,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle.clone(), true)?;
        let table = orders_table(nullability)?;
        let file = write(&session, &table).await?;

        let id = get_item("id", root());
        let filter = list_contains(item_field("sku")?, lit("sku-4001-1"));
        let (actual, filter_read) =
            scan_counting(&file, executor, handle.clone(), &id, Some(&filter)).await?;

        let mut ctx = session.create_execution_ctx();
        let keep = table
            .clone()
            .apply(&fill_null(filter, lit(false)))?
            .execute::<Mask>(&mut ctx)?;
        assert_eq!(keep.true_count(), 1);
        assert_arrays_eq!(actual, table.clone().filter(keep)?.apply(&id)?, &mut ctx);

        let ids_and_skus = pack(
            [("id", id), ("skus", item_field("sku")?)],
            Nullability::NonNullable,
        );
        let (_, projection_read) =
            scan_counting(&file, executor, handle, &ids_and_skus, None).await?;
        assert!(filter_read.segments_requested <= projection_read.segments_requested);
        Ok(())
    })
}
