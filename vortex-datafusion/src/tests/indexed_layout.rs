// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test that DataFusion can query a file whose column uses the `vortex.indexed` layout.

use std::sync::Arc;

use arrow_array::record_batch;
use datafusion::arrow::array::RecordBatch;
use datafusion::assert_batches_sorted_eq;
use datafusion::physical_plan::collect;
use datafusion_expr::col;
use datafusion_expr::lit;
use datafusion_physical_plan::metrics::MetricsSet;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::dtype::FieldPath;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::indexed::INDEXED_LAYOUT_ID;
use vortex::layout::layouts::indexed::IndexConfig;
use vortex::layout::layouts::indexed::IndexSessionExt;
use vortex::layout::layouts::indexed::IndexedStrategy;
use vortex::layout::layouts::repartition::RepartitionStrategy;
use vortex::layout::layouts::repartition::RepartitionWriterOptions;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_reverse_index::ReverseIndex;

use crate::VortexFormatFactory;
use crate::VortexTableOptions;
use crate::common_tests::TestSessionContext;
use crate::metrics::VortexMetricsFinder;

const VALUE_FIELD: &str = "value";
/// Small enough that the 12-row batch spans three blocks under the indexed layout, leaving the
/// index a real pruning decision to make instead of degenerating to a single block.
const BLOCK_LEN: usize = 4;

/// The default session doesn't enable `vortex.indexed` for writing — it's a layout prototype, not
/// part of any frozen `core` edition — so this test registers a tiny edition just for it.
const INDEXED_TEST_EDITION: EditionId =
    EditionId::new("vortex-datafusion-indexed-test", 2026, 1, 0);

static INDEXED_TEST_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: INDEXED_TEST_EDITION,
        min_vortex_version: None,
    },
    added: &[EditionMember::layout(&INDEXED_LAYOUT_ID)],
};

/// A session that can write and read the `vortex.indexed` layout, with a [`ReverseIndex`]
/// registered as an index kind.
fn session_with_indexed_layout() -> anyhow::Result<VortexSession> {
    let session = VortexSession::default();
    session.register_edition(&INDEXED_TEST_DECLARATION)?;
    session.enable_edition(INDEXED_TEST_EDITION)?;
    session.indexes().register(ReverseIndex::new_ref());
    Ok(session)
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

    let opts = VortexTableOptions {
        projection_pushdown,
        ..Default::default()
    };
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
/// [`InstrumentedReadAt`](vortex::io::VortexReadAt)'s `vortex.io.read.total_size` counter.
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
    let opts = VortexTableOptions {
        projection_pushdown,
        ..Default::default()
    };
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
        VortexSession::default(),
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
