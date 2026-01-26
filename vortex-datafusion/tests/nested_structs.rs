// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::Field;
use arrow_schema::Fields;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::buffer::NullBuffer;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::execution::SessionStateBuilder;
use datafusion::parquet::arrow::AsyncArrowWriter;
use datafusion::parquet::arrow::async_writer::ParquetObjectWriter;
use datafusion::prelude::ParquetReadOptions;
use datafusion::prelude::SessionContext;
use datafusion_common::assert_batches_sorted_eq;
use datafusion_common::create_array;
use datafusion_datasource::ListingTableUrl;
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
    let array = ArrayRef::from_arrow(records, false).unwrap();
    let path = Path::from_url_path(path).unwrap();
    let mut write = ObjectStoreWriter::new(store.clone(), &path).await.unwrap();
    SESSION
        .write_options()
        .write(&mut write, array.to_array_stream())
        .await
        .unwrap();
    write.shutdown().await.unwrap();
}

async fn write_parquet(store: &Arc<dyn ObjectStore>, path: &str, records: &RecordBatch) {
    let w = ParquetObjectWriter::new(Arc::clone(store), Path::from_url_path(path).unwrap());
    let mut async_writer =
        AsyncArrowWriter::try_new(w, Arc::clone(records.schema_ref()), None).unwrap();
    async_writer.write(records).await.unwrap();
    async_writer.close().await.unwrap();
}

#[tokio::test]
pub async fn scan_nested_struct() {
    let (ctx, store) = make_session_ctx();

    // Nested structs with nullable fields projection.
    // We test that this has the same behavior as using the same data as Parquet.

    let c: ArrowArrayRef = create_array!(Int32, vec![1, 2, 3, 4]);
    let b: ArrowArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![Field::new("c", c.data_type().clone(), false)]),
        vec![c],
        None,
    ));

    let a_nulls = NullBuffer::from_iter([true, true, false, true]);

    let a: ArrowArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![Field::new("b", b.data_type().clone(), false)]),
        vec![b],
        Some(a_nulls),
    ));

    // Write a new record batch of the nested data to a Vortex file
    let batch = RecordBatch::try_from_iter(vec![("a", a)]).expect("record batch construction");

    // Write as Vortex and as Parquet so we can compare them
    write_file(&store, "vortex/file.vortex", &batch).await;
    write_parquet(&store, "parquet/file.parquet", &batch).await;

    let expected = [
        "+------------------------+",
        "| nested_parquet.a[b][c] |",
        "+------------------------+",
        "| 1                      |",
        "| 2                      |",
        "| 4                      |",
        "| NULL                   |",
        "+------------------------+",
    ];

    ctx.register_parquet(
        "nested_parquet",
        "s3://in-memory/parquet/",
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let memory_results = ctx
        .sql("select a.b.c from nested_parquet")
        .await
        .expect("query nested_parquet")
        .collect()
        .await
        .unwrap();

    // assert_batches_sorted_eq!(expected, &memory_results);

    // Load Vortex table
    ctx.register_listing_table(
        "nested_vortex",
        "s3://in-memory/vortex/",
        ListingOptions::new(Arc::new(VortexFormat::new(SESSION.clone())))
            .with_session_config_options(ctx.state().config())
            .with_file_extension("vortex"),
        None,
        None,
    )
    .await
    .unwrap();

    let vortex_results = ctx
        .sql("select a.b.c from nested_vortex")
        .await
        .expect("query nested_vortex")
        .collect()
        .await
        .unwrap();

    let expected = [
        "+------------------------+",
        "| nested_vortex.a[b][c] |",
        "+------------------------+",
        "| 1                     |",
        "| 2                     |",
        "| 4                     |",
        "| NULL                  |",
        "+-----------------------+",
    ];
    assert_batches_sorted_eq!(expected, &vortex_results);
}
