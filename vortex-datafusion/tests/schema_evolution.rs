// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::unwrap_in_result,
    clippy::unwrap_used,
    clippy::tests_outside_test_module
)]

//! Test that checks we can evolve schemas in a compatible way across files.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::datatypes::UInt16Type;
use datafusion::arrow::datatypes::UInt32Type;
use datafusion::assert_batches_sorted_eq;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionContext;
use datafusion_common::assert_batches_eq;
use datafusion_common::create_array;
use datafusion_common::record_batch;
use datafusion_datasource::ListingTableUrl;
use datafusion_expr::col;
use datafusion_expr::lit;
use datafusion_functions::expr_fn::get_field;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::ObjectStoreWriter;
use vortex::io::VortexWrite;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexFormatFactory;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

fn register_vortex_format_factory(
    factory: VortexFormatFactory,
    session_state_builder: &mut SessionStateBuilder,
) {
    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            datafusion::common::GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            Arc::new(datafusion::datasource::provider::DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(Arc::new(factory));
    }
}

fn make_session_ctx() -> (SessionContext, Arc<dyn ObjectStore>) {
    let factory: VortexFormatFactory = VortexFormatFactory::new();
    let mut session_state_builder = SessionStateBuilder::new().with_default_features();
    register_vortex_format_factory(factory, &mut session_state_builder);
    let ctx = SessionContext::new_with_state(session_state_builder.build());
    let store = Arc::new(InMemory::new());
    ctx.register_object_store(&Url::parse("s3://in-memory/").unwrap(), store.clone());

    (ctx, store)
}

async fn write_file(store: &Arc<dyn ObjectStore>, path: &str, records: &RecordBatch) {
    let array = ArrayRef::from_arrow(records, false);
    let path = Path::from_url_path(path).unwrap();
    let mut write = ObjectStoreWriter::new(store.clone(), &path).await.unwrap();
    SESSION
        .write_options()
        .write(&mut write, array.to_array_stream())
        .await
        .unwrap();
    write.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_filter_with_schema_evolution() {
    let (ctx, store) = make_session_ctx();

    // file1 only contains field "a"
    write_file(
        &store,
        "files/file1.vortex",
        &record_batch!(("a", Utf8, vec![Some("one"), Some("two"), Some("three")])).unwrap(),
    )
    .await;

    // file2 only contains field "b"
    write_file(
        &store,
        "files/file2.vortex",
        &record_batch!(("b", Utf8, vec![Some("four"), Some("five"), Some("six")])).unwrap(),
    )
    .await;

    // Read the table back as Vortex
    let table_url = ListingTableUrl::parse("s3://in-memory/files").unwrap();
    let list_opts = ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
        .with_session_config_options(ctx.state().config())
        .with_file_extension("vortex");

    let table = ListingTable::try_new(
        ListingTableConfig::new(table_url)
            .with_listing_options(list_opts)
            .infer_schema(&ctx.state())
            .await
            .unwrap(),
    )
    .unwrap();

    let table = Arc::new(table);

    let df = ctx.read_table(table).unwrap();

    let table_schema = Arc::new(df.schema().as_arrow().clone());

    // Table schema contains both fields
    assert_eq!(
        table_schema.as_ref(),
        &Schema::new(vec![
            Field::new("a", DataType::Utf8View, true),
            Field::new("b", DataType::Utf8View, true),
        ])
    );

    // Filter the result to only ones with a column, i.e. only file1
    let result = df
        .filter(col("a").is_not_null())
        .unwrap()
        .collect()
        .await
        .unwrap();

    let expected = [
        "+-------+---+",
        "| a     | b |",
        "+-------+---+",
        "| one   |   |",
        "| three |   |",
        "| two   |   |",
        "+-------+---+",
    ];
    assert_batches_sorted_eq!(expected, &result);
}

#[tokio::test]
async fn test_filter_schema_evolution_order() {
    let (ctx, store) = make_session_ctx();

    // file1 only contains field "a"
    write_file(
        &store,
        "files/file1.vortex",
        &record_batch!(("a", Int32, vec![Some(1), Some(3), Some(5)])).unwrap(),
    )
    .await;

    // file2 containing fields "b" and "a", where "a" needs to be upcast at scan time.
    write_file(
        &store,
        "files/file2.vortex",
        &record_batch!(
            ("b", Utf8, vec![Some("two"), Some("four"), Some("six")]),
            ("a", Int16, vec![Some(2), Some(4), Some(6)])
        )
        .unwrap(),
    )
    .await;

    // Read the table back as Vortex
    let table_url = ListingTableUrl::parse("s3://in-memory/files").unwrap();
    let list_opts = ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
        .with_session_config_options(ctx.state().config())
        .with_file_extension("vortex");

    // We force the table schema, because file1/file2 have different types for the "a" column
    let read_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8View, true),
    ]));

    let table = ListingTable::try_new(
        ListingTableConfig::new(table_url)
            .with_listing_options(list_opts)
            .with_schema(read_schema.clone()),
    )
    .unwrap();

    let table = Arc::new(table);

    let df = ctx.read_table(table.clone()).unwrap();

    let table_schema = Arc::new(df.schema().as_arrow().clone());

    // Table schema contains both fields
    assert_eq!(
        table_schema.as_ref(),
        &Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ])
    );

    // Filter referencing the b column, which only appears in file2
    let result = df
        .filter(col("b").eq(lit("two")))
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_eq!(
        &[
            "+---+-----+",
            "| a | b   |",
            "+---+-----+",
            "| 2 | two |",
            "+---+-----+",
        ],
        &result
    );

    // Filter on the "a" column, which has different types for each file
    let result = ctx
        .read_table(table)
        .unwrap()
        .filter(col("a").gt_eq(lit(3i16)))
        .unwrap()
        .collect()
        .await
        .unwrap();
    // let table = concat_batches(&table_schema, result.iter()).unwrap();

    // a field: present in both files
    // b field: only present in file2, file1 fills with nulls
    assert_batches_sorted_eq!(
        &[
            "+---+------+",
            "| a | b    |",
            "+---+------+",
            "| 3 |      |",
            "| 4 | four |",
            "| 5 |      |",
            "| 6 | six  |",
            "+---+------+",
        ],
        &result
    );
}

#[tokio::test]
async fn test_filter_schema_evolution_struct_fields() {
    // Test for correct schema evolution behavior in the presence of nested struct fields.
    // We use a hypothetical schema of some observability data with "wide records", struct columns
    // with nullable payloads that may or may not be present for every file.

    let (ctx, store) = make_session_ctx();

    fn make_metrics(
        hostname: &str,
        uptime: Vec<i64>,
        instance: Option<Vec<Option<&str>>>,
    ) -> RecordBatch {
        let values_array: ArrowArrayRef = create_array!(Int64, uptime);
        let payload_array = if let Some(tags) = instance {
            let tags_array: ArrowArrayRef = create_array!(Utf8, tags);
            Arc::new(StructArray::new(
                vec![
                    Field::new("uptime", DataType::Int64, true),
                    Field::new("instance", DataType::Utf8, true),
                ]
                .into(),
                vec![values_array, tags_array],
                None,
            ))
        } else {
            Arc::new(StructArray::new(
                vec![Field::new("uptime", DataType::Int64, true)].into(),
                vec![values_array],
                None,
            ))
        };

        let len = payload_array.len();
        let hostname_array = create_array!(Utf8, vec![Some(hostname); len]);

        let payload_type = payload_array.data_type().clone();
        let hostname_type = hostname_array.data_type().clone();

        RecordBatch::from(StructArray::new(
            vec![
                Field::new("hostname", hostname_type, true),
                Field::new("payload", payload_type, true),
            ]
            .into(),
            vec![hostname_array, payload_array],
            None,
        ))
    }

    let host01 = make_metrics("host01.local", vec![1, 2, 3, 4], None);
    let host02 = make_metrics(
        "host02.local",
        vec![10, 20, 30, 40],
        // host02 has new logging code which adds the new "instance" nested field in its payload
        Some(vec![Some("c6i"), Some("c6i"), Some("m5"), Some("r5")]),
    );

    // Write metrics files to storage
    write_file(&store, "files/host01.vortex", &host01).await;
    write_file(&store, "files/host02.vortex", &host02).await;

    // Read the table back as Vortex
    let table_url = ListingTableUrl::parse("s3://in-memory/files").unwrap();
    let list_opts = ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
        .with_session_config_options(ctx.state().config())
        .with_file_extension("vortex");

    // We force the table schema to be the one inclusive of the new instance field.
    let read_schema = host02.schema();

    let table = ListingTable::try_new(
        ListingTableConfig::new(table_url)
            .with_listing_options(list_opts)
            .with_schema(read_schema.clone()),
    )
    .unwrap();

    let table = Arc::new(table);

    let df = ctx.read_table(table.clone()).unwrap();

    let table_schema = Arc::new(df.schema().as_arrow().clone());

    // Table schema contains both fields
    assert_eq!(table_schema.as_ref(), read_schema.as_ref(),);

    // Scan all the records, NULLs are filled in for nested optional fields.
    let full_scan = df.collect().await.unwrap();

    assert_batches_sorted_eq!(
        &[
            "+--------------+-----------------------------+",
            "| hostname     | payload                     |",
            "+--------------+-----------------------------+",
            "| host01.local | {uptime: 1, instance: }     |",
            "| host01.local | {uptime: 2, instance: }     |",
            "| host01.local | {uptime: 3, instance: }     |",
            "| host01.local | {uptime: 4, instance: }     |",
            "| host02.local | {uptime: 10, instance: c6i} |",
            "| host02.local | {uptime: 20, instance: c6i} |",
            "| host02.local | {uptime: 30, instance: m5}  |",
            "| host02.local | {uptime: 40, instance: r5}  |",
            "+--------------+-----------------------------+",
        ],
        &full_scan
    );

    // run a filter that touches both the payload.uptime AND the payload.instance nested fields
    let df = ctx.read_table(table.clone()).unwrap();
    let filtered_scan = df
        .filter(
            // payload.instance = 'c6i' OR payload.uptime < 10
            // We need to perform filtering over nested columns which don't exist in every
            // file type.
            get_field(col("payload"), "instance")
                .eq(lit("c6i"))
                .or(get_field(col("payload"), "uptime").lt(lit(10))),
        )
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_batches_sorted_eq!(
        &[
            "+--------------+-----------------------------+",
            "| hostname     | payload                     |",
            "+--------------+-----------------------------+",
            "| host01.local | {uptime: 1, instance: }     |",
            "| host01.local | {uptime: 2, instance: }     |",
            "| host01.local | {uptime: 3, instance: }     |",
            "| host01.local | {uptime: 4, instance: }     |",
            "| host02.local | {uptime: 10, instance: c6i} |",
            "| host02.local | {uptime: 20, instance: c6i} |",
            "+--------------+-----------------------------+",
        ],
        &filtered_scan
    );
}

#[tokio::test]
async fn test_schema_evolution_struct_of_dict() -> anyhow::Result<()> {
    let (ctx, store) = make_session_ctx();

    // First file
    let struct_fields = Fields::from(vec![
        Field::new_dictionary("a", DataType::UInt16, DataType::Utf8, true),
        Field::new_dictionary("b", DataType::UInt16, DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(DictionaryArray::<UInt16Type>::from_iter(["x1", "y1", "x1"])),
            Arc::new(DictionaryArray::<UInt16Type>::from_iter(["p1", "p1", "q1"])),
        ],
        None,
    );

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(struct_fields),
            true,
        )])),
        vec![Arc::new(struct_array)],
    )?;

    write_file(&store, "files/file1.vortex", &batch).await;

    // Second file
    let struct_fields = Fields::from(vec![
        Field::new_dictionary("a", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("b", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("c", DataType::UInt32, DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["x2", "y2", "x2"])),
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["p2", "p2", "q2"])),
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["a2", "b2", "c2"])),
        ],
        None,
    );

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(struct_fields.clone()),
            true,
        )])),
        vec![Arc::new(struct_array)],
    )?;

    write_file(&store, "files/file2.vortex", &batch).await;

    let read_schema = batch.schema();

    // Read the table back as Vortex
    let table_url = ListingTableUrl::parse("s3://in-memory/files").unwrap();
    let list_opts = ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
        .with_session_config_options(ctx.state().config())
        .with_file_extension("vortex");

    let table = Arc::new(ListingTable::try_new(
        ListingTableConfig::new(table_url)
            .with_listing_options(list_opts)
            .with_schema(read_schema.clone()),
    )?);

    let df = ctx.read_table(table.clone()).unwrap();
    let table_schema = Arc::new(df.schema().as_arrow().clone());

    assert_eq!(table_schema.as_ref(), read_schema.as_ref());

    let full_scan = df.collect().await.unwrap();

    assert_batches_sorted_eq!(
        &[
            "+-----------------------+",
            "| my_struct             |",
            "+-----------------------+",
            "| {a: x1, b: p1, c: }   |",
            "| {a: x1, b: q1, c: }   |",
            "| {a: x2, b: p2, c: a2} |",
            "| {a: x2, b: q2, c: c2} |",
            "| {a: y1, b: p1, c: }   |",
            "| {a: y2, b: p2, c: b2} |",
            "+-----------------------+",
        ],
        &full_scan
    );

    let filter =
        get_field(col("my_struct"), "a")
            .eq(lit("x1"))
            .or(get_field(col("my_struct"), "a").eq(lit("x2")));
    // run a filter that touches both the payload.uptime AND the payload.instance nested fields
    let df = ctx.read_table(table.clone())?;
    let filtered_scan = df.filter(filter)?.collect().await?;

    assert_eq!(filtered_scan[0].schema(), read_schema);

    assert_batches_sorted_eq!(
        &[
            "+-----------------------+",
            "| my_struct             |",
            "+-----------------------+",
            "| {a: x1, b: p1, c: }   |",
            "| {a: x1, b: q1, c: }   |",
            "| {a: x2, b: p2, c: a2} |",
            "| {a: x2, b: q2, c: c2} |",
            "+-----------------------+",
        ],
        &filtered_scan
    );

    Ok(())
}
