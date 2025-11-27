// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::unwrap_in_result,
    clippy::unwrap_used,
    clippy::tests_outside_test_module
)]

//! Test that checks we can evolve schemas in a cmpatible way across files.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion::arrow::array::StringViewArray;
use datafusion::arrow::compute::concat_batches;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionContext;
use datafusion_common::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::record_batch;
use datafusion_datasource::ListingTableUrl;
use datafusion_expr::col;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use url::Url;
use vortex::ArrayRef;
use vortex::VortexSessionDefault;
use vortex::arrow::FromArrowArray;
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
    let table = concat_batches(&table_schema, result.iter()).unwrap();

    // We read back the full table, with nulls filled in for missing fields
    assert_eq!(
        table,
        record_batch(
            &table_schema,
            vec![
                // a
                Arc::new(StringViewArray::from(vec![
                    Some("one"),
                    Some("two"),
                    Some("three"),
                ])) as ArrowArrayRef,
                // b
                Arc::new(StringViewArray::from(vec![
                    Option::<&str>::None,
                    None,
                    None
                ])) as ArrowArrayRef,
            ]
        )
    );
}

fn record_batch(
    schema: &SchemaRef,
    fields: impl IntoIterator<Item = ArrowArrayRef>,
) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), fields.into_iter().collect()).unwrap()
}
