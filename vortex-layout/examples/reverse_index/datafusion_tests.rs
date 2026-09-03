//! Test that DataFusion can query a file whose column uses the `vortex.indexed` layout, with
//! [`ReverseIndex`] answering an equality filter.
//!
//! This lives alongside the [`ReverseIndex`] example rather than in `vortex-datafusion` because
//! `vortex-layout` sits below `vortex-datafusion` in the dependency graph: `vortex-datafusion`
//! cannot depend on an example that lives in `vortex-layout`, but an example's `dev-dependencies`
//! may freely depend on `vortex-datafusion`.

use std::sync::Arc;

use arrow_array::record_batch;
use datafusion::arrow::array::RecordBatch;
use datafusion::assert_batches_sorted_eq;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_catalog::TableProvider;
use datafusion_common::DFSchema;
use datafusion_common::GetExt;
use datafusion_expr::CreateExternalTable;
use datafusion_expr::col;
use datafusion_expr::lit;
use datafusion_physical_plan::metrics::MetricsSet;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use url::Url;
use vortex_array::array_session;
use vortex_array::dtype::FieldPath;
use vortex_arrow::ArrowSessionExt;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;
use vortex_datafusion::metrics::VortexMetricsFinder;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::VortexWrite;
use vortex_io::object_store::ObjectStoreWrite;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::INDEXED_LAYOUT_ID;
use vortex_layout::layouts::indexed::IndexConfig;
use vortex_layout::layouts::indexed::IndexSessionExt;
use vortex_layout::layouts::indexed::IndexedStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::ReverseIndex;

const VALUE_FIELD: &str = "value";
/// Small enough that the 12-row batch spans three blocks under the indexed layout, leaving the
/// index a real pruning decision to make instead of degenerating to a single block.
const BLOCK_LEN: usize = 4;

/// The array/layout encodings this test needs to write, converted from Arrow via
/// [`vortex_arrow::ArrowSessionExt`].
///
/// The default Vortex file writer only permits array/layout ids covered by the session's enabled
/// editions, but those first-party declarations live in the `vortex` facade crate, which
/// `vortex-layout` cannot depend on. Declaring and enabling a tiny edition here is the local
/// equivalent.
const INDEXED_TEST_EDITION: EditionId =
    EditionId::new("vortex-layout-reverse-index-datafusion-test", 2026, 1, 0);

static INDEXED_TEST_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: INDEXED_TEST_EDITION,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.struct"),
        EditionMember::array(&"vortex.primitive"),
        // The index child's postings column is serialized roaring bitmaps, written as varbinview.
        EditionMember::array(&"vortex.varbinview"),
        // `payload`'s sequential row-index values compress with the sequence encoding; `value`'s
        // single repeated value per block compresses with the constant encoding.
        EditionMember::array(&"vortex.sequence"),
        EditionMember::array(&"vortex.constant"),
        EditionMember::layout(&"vortex.struct"),
        EditionMember::layout(&"vortex.chunked"),
        EditionMember::layout(&"vortex.flat"),
        EditionMember::layout(&"vortex.zoned"),
        EditionMember::layout(&INDEXED_LAYOUT_ID),
        // Zone-map stats over each chunk need min/max/null-count, computed as aggregates during
        // writing.
        EditionMember::aggregate(&"vortex.min"),
        EditionMember::aggregate(&"vortex.max"),
        EditionMember::aggregate(&"vortex.null_count"),
    ],
};

/// A session that can write and read the `vortex.indexed` layout, with a [`ReverseIndex`]
/// registered as an index kind.
///
/// [`array_session`] already bundles everything [`vortex::VortexSessionDefault::default`] would
/// (arrays, dtypes, scalar functions, stats, optimizer kernels, aggregate functions, and memory);
/// this only adds the layout, runtime, and edition state that live in higher-level crates.
fn session_with_indexed_layout() -> anyhow::Result<VortexSession> {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<EditionSession>();
    vortex_arrow::initialize(&session);
    vortex_sequence::initialize(&session);
    session.register_edition(&INDEXED_TEST_DECLARATION)?;
    session.enable_edition(INDEXED_TEST_EDITION)?;
    session.indexes().register(ReverseIndex::new_ref());
    Ok(session)
}

/// A session that can read the `vortex.indexed` layout, but without a [`ReverseIndex`]
/// registered — the fallback path `IndexedReader::plan_probe` takes when an index kind's spec
/// goes unclaimed.
fn session_without_reverse_index() -> VortexSession {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<EditionSession>();
    vortex_arrow::initialize(&session);
    vortex_sequence::initialize(&session);
    session
}

/// A write strategy that attaches a [`ReverseIndex`] to the `value` field, chunked into blocks of
/// `block_len` rows.
fn indexed_write_strategy(block_len: usize) -> Arc<dyn LayoutStrategy> {
    let data = RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: block_len,
            block_size_target: None,
            canonicalize: false,
        },
    );
    let indexed = IndexedStrategy::new(
        data,
        FlatLayoutStrategy::default(),
        vec![IndexConfig::with_defaults(ReverseIndex::new_ref())],
    )
    .with_data_block_len(block_len as u64);

    WriteStrategyBuilder::default()
        .with_row_block_size(block_len)
        .with_field_writer(FieldPath::from_name(VALUE_FIELD), Arc::new(indexed))
        .build()
}

/// `20` repeats at rows 1 and 9; `999` never appears. With [`BLOCK_LEN`] of 4 those land in
/// blocks 0 and 2, leaving block 1 fully prunable by an exact equality index.
fn test_batch() -> anyhow::Result<RecordBatch> {
    Ok(record_batch!((
        "value",
        Int32,
        vec![
            Some(10),
            Some(20),
            Some(30),
            Some(40),
            Some(50),
            Some(60),
            Some(70),
            Some(80),
            Some(90),
            Some(20),
            Some(100),
            Some(110)
        ]
    ))?)
}

/// A minimal DataFusion harness over an in-memory [`ObjectStore`], built only from
/// `vortex-datafusion`'s public API (the crate's own richer `TestSessionContext` is `#[cfg(test)]`
/// only, so it isn't visible outside `vortex-datafusion` itself).
struct TestSessionContext {
    store: Arc<dyn ObjectStore>,
    session: SessionContext,
}

impl TestSessionContext {
    fn new_with_factory(factory: Arc<VortexFormatFactory>) -> Self {
        let store = Arc::new(InMemory::new());
        let mut session_state_builder = SessionStateBuilder::new()
            .with_default_features()
            .with_table_factory(
                factory.get_ext().to_uppercase(),
                Arc::new(DefaultTableFactory::new()),
            )
            .with_object_store(
                &Url::try_from("file://").unwrap(),
                Arc::<InMemory>::clone(&store),
            );

        if let Some(file_formats) = session_state_builder.file_formats() {
            file_formats.push(factory as _);
        }

        let session =
            SessionContext::new_with_state(session_state_builder.build()).enable_url_table();

        Self { store, session }
    }

    async fn table_provider<S>(
        &self,
        name: &str,
        location: impl Into<String>,
        schema: S,
    ) -> anyhow::Result<Arc<dyn TableProvider>>
    where
        DFSchema: TryFrom<S>,
        anyhow::Error: From<<S as TryInto<DFSchema>>::Error>,
    {
        let factory = self.session.table_factory("VORTEX").unwrap();

        let cmd = CreateExternalTable::builder(
            name,
            location.into(),
            "vortex",
            DFSchema::try_from(schema)?.into(),
        )
        .build();

        let table = factory.create(&self.session.state(), &cmd).await?;

        Ok(table)
    }
}

async fn write_indexed_batch(
    ctx: &TestSessionContext,
    session: &VortexSession,
    path: &str,
    batch: &RecordBatch,
    block_len: usize,
) -> anyhow::Result<()> {
    let array = session
        .arrow()
        .from_arrow_record_batch(batch.clone(), &batch.schema())?;
    let mut write = ObjectStoreWrite::new(Arc::clone(&ctx.store), &path.into()).await?;
    session
        .write_options()
        .with_strategy(indexed_write_strategy(block_len))
        .write(&mut write, array.to_array_stream())
        .await?;
    write.shutdown().await?;
    Ok(())
}

/// `20` repeats at rows 1 and 9; `999` never appears. An equality filter on `value` is exactly
/// what [`ReverseIndex::plan`](vortex_layout::layouts::indexed::IndexVTable::plan) claims, so this
/// exercises both "claimed and found" and "claimed but no match" through DataFusion's predicate
/// pushdown into the indexed column.
#[rstest]
#[tokio::test]
async fn test_query_over_indexed_column(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let session = session_with_indexed_layout()?;

    let mut opts = VortexTableOptions::default();
    opts.projection_pushdown = projection_pushdown;
    let factory = Arc::new(VortexFormatFactory::new_with_options(session.clone(), opts));
    let ctx = TestSessionContext::new_with_factory(factory);

    let batch = test_batch()?;
    write_indexed_batch(&ctx, &session, "files/indexed.vortex", &batch, BLOCK_LEN).await?;

    let schema = batch.schema();
    let provider = ctx
        .table_provider("indexed_tbl", "/files/", schema.as_ref().clone())
        .await?;
    let table = ctx.session.read_table(provider)?;

    let matches = table
        .clone()
        .filter(col(VALUE_FIELD).eq(lit(20)))?
        .collect()
        .await?;
    assert_batches_sorted_eq!(
        [
            "+-------+",
            "| value |",
            "+-------+",
            "| 20    |",
            "| 20    |",
            "+-------+",
        ],
        &matches
    );

    let absent = table
        .filter(col(VALUE_FIELD).eq(lit(999)))?
        .collect()
        .await?;
    assert!(absent.iter().all(|batch| batch.num_rows() == 0));

    Ok(())
}

/// Rows per block for [`pruning_test_batch`], and the number of blocks it spans.
const PRUNING_BLOCK_LEN: usize = 100;
const PRUNING_BLOCK_COUNT: usize = 5;
const PAYLOAD_FIELD: &str = "payload";

/// `value` is clustered by block: block `k` (rows `k * PRUNING_BLOCK_LEN` to
/// `(k + 1) * PRUNING_BLOCK_LEN - 1`) holds nothing but the constant `k + 1`. Filtering on a
/// single value therefore claims exactly one block as a match and leaves every other block fully
/// prunable, unlike [`test_batch`]'s handful of rows, where the pruned savings are too small to
/// stand out over the index's own storage overhead.
///
/// `payload` carries the row index and is neither indexed nor filtered on — selecting it instead
/// of `value` means the data child only needs the rows the index's mask actually claims, rather
/// than every row `value` itself requires to be decoded and re-checked against the filter.
fn pruning_test_batch() -> anyhow::Result<RecordBatch> {
    let row_count = PRUNING_BLOCK_LEN * PRUNING_BLOCK_COUNT;
    let mut values: Vec<Option<i32>> = Vec::with_capacity(row_count);
    for block in 0..PRUNING_BLOCK_COUNT {
        let value = i32::try_from(block)? + 1;
        values.extend(std::iter::repeat_n(Some(value), PRUNING_BLOCK_LEN));
    }
    let payload: Vec<Option<i32>> = (0..row_count)
        .map(|row| i32::try_from(row).map(Some))
        .collect::<Result<_, _>>()?;

    Ok(record_batch!(
        ("value", Int32, values),
        ("payload", Int32, payload)
    )?)
}

/// Total bytes read from storage across every Vortex-backed data source in the plan, per
/// `InstrumentedReadAt`'s `vortex.io.read.total_size` counter.
fn total_bytes_read(metrics_sets: &[MetricsSet]) -> usize {
    metrics_sets
        .iter()
        .filter_map(|set| set.sum_by_name("vortex.io.read.total_size"))
        .map(|value| value.as_usize())
        .sum()
}

/// Runs `value = <target>` against a freshly written copy of `batch`, executing the physical plan
/// directly (rather than through [`DataFrame::collect`]) so the same plan instance can be
/// inspected for metrics afterward.
///
/// [`DataFrame::collect`]: datafusion::dataframe::DataFrame::collect
async fn run_equality_filter(
    write_session: &VortexSession,
    read_session: VortexSession,
    projection_pushdown: bool,
    batch: &RecordBatch,
    block_len: usize,
    target: i32,
) -> anyhow::Result<(Vec<RecordBatch>, usize)> {
    let mut opts = VortexTableOptions::default();
    opts.projection_pushdown = projection_pushdown;
    let factory = Arc::new(VortexFormatFactory::new_with_options(read_session, opts));
    let ctx = TestSessionContext::new_with_factory(factory);

    write_indexed_batch(
        &ctx,
        write_session,
        "files/indexed.vortex",
        batch,
        block_len,
    )
    .await?;

    let schema = batch.schema();
    let provider = ctx
        .table_provider("indexed_tbl", "/files/", schema.as_ref().clone())
        .await?;
    ctx.session.register_table("indexed_tbl", provider)?;

    let df = ctx
        .session
        .sql(&format!(
            "SELECT {PAYLOAD_FIELD} FROM indexed_tbl WHERE {VALUE_FIELD} = {target}"
        ))
        .await?;
    let physical_plan = ctx
        .session
        .state()
        .create_physical_plan(df.logical_plan())
        .await?;
    let results = collect(Arc::clone(&physical_plan), ctx.session.task_ctx()).await?;
    let bytes_read = total_bytes_read(&VortexMetricsFinder::find_all(physical_plan.as_ref()));

    Ok((results, bytes_read))
}

/// Correctness alone can't distinguish "the index answered the filter exactly" from "the index
/// was silently ignored and a full scan happened to get the right answer anyway" — both produce
/// identical query results. This test tells them apart by comparing bytes read from storage for
/// the same file and filter, once with [`ReverseIndex`] registered for reading and once without
/// (which forces the documented fallback: `IndexedReader::plan_probe` treats an unregistered
/// index kind's spec as inert). [`pruning_test_batch`] puts a single value in each block, so a
/// real exact-index probe lets the scan skip every block but one, while the unregistered run must
/// decode all of them to filter.
#[rstest]
#[tokio::test]
async fn test_index_avoids_reading_pruned_blocks(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let write_session = session_with_indexed_layout()?;
    let batch = pruning_test_batch()?;
    // Block 0, not a middle block: object_store's read coalescing merges nearby byte ranges into
    // one physical read, so a "hole" in the middle of the file still gets pulled in as padding. A
    // match confined to the first block is a genuine prefix, so skipping the rest of the file
    // actually shrinks the range requested.
    let target = 1;

    let (with_index_rows, with_index_bytes) = run_equality_filter(
        &write_session,
        session_with_indexed_layout()?,
        projection_pushdown,
        &batch,
        PRUNING_BLOCK_LEN,
        target,
    )
    .await?;
    let (without_index_rows, without_index_bytes) = run_equality_filter(
        &write_session,
        session_without_reverse_index(),
        projection_pushdown,
        &batch,
        PRUNING_BLOCK_LEN,
        target,
    )
    .await?;

    assert_eq!(with_index_rows, without_index_rows);
    let matched_rows: usize = with_index_rows.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(matched_rows, PRUNING_BLOCK_LEN);

    assert!(
        with_index_bytes < without_index_bytes,
        "expected the registered index to skip the prunable blocks and read fewer bytes than the \
         unregistered fallback, got {with_index_bytes} (indexed) vs {without_index_bytes} \
         (fallback)"
    );

    Ok(())
}
