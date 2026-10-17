//! Verifies that a registered index actually skips reading pruned blocks, not just that it
//! returns the right rows.
//!
//! This scans directly through [`vortex_layout`]'s `ScanBuilder` against an in-memory
//! [`ObjectStore`], rather than through `vortex-datafusion`: DataFusion's own equality-pushdown
//! behavior is already covered by `vortex-datafusion`'s own test suite, so re-proving it here only
//! bought this crate's tests a `datafusion` dev-dependency and a much slower compile.

use std::num::NonZeroU64;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::record_batch;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::InstrumentedReadAt;
use vortex_io::VortexWrite;
use vortex_io::object_store::ObjectStoreReadAt;
use vortex_io::object_store::ObjectStoreWrite;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
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
use vortex_metrics::DefaultMetricsRegistry;
use vortex_metrics::MetricValue;
use vortex_metrics::MetricsRegistry;
use vortex_session::VortexSession;

use crate::ReverseIndex;

const VALUE_FIELD: &str = "value";

/// The array/layout encodings this test needs to write, converted from Arrow via
/// [`vortex_arrow::ArrowSessionExt`].
///
/// The default Vortex file writer only permits array/layout ids covered by the session's enabled
/// editions, but those first-party declarations live in the `vortex` facade crate, which
/// `vortex-layout` cannot depend on. Declaring and enabling a tiny edition here is the local
/// equivalent.
const INDEXED_TEST_EDITION: EditionId =
    EditionId::new("vortex-layout-reverse-index-pruning-test", 2026, 1, 0);

static INDEXED_TEST_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: INDEXED_TEST_EDITION,
        min_library_version: None,
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
///
/// With a `partition_len`, the index is built per partition and each partition is written as its
/// own index-child chunk, so a probe only fetches the partitions a split overlaps.
fn indexed_write_strategy(
    session: &VortexSession,
    block_len: usize,
    partition_len: Option<NonZeroU64>,
) -> Arc<dyn LayoutStrategy> {
    let data = RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: block_len,
            block_size_target: None,
            canonicalize: false,
        },
    );
    let mut config = IndexConfig::with_defaults(ReverseIndex::new_ref());
    if let Some(partition_len) = partition_len {
        config = config.with_partition_len(partition_len);
    }
    let indexed = IndexedStrategy::new(
        data,
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        vec![config],
    )
    .with_data_block_len(block_len as u64);

    WriteStrategyBuilder::from_session(session)
        .with_row_block_size(block_len)
        .with_field_writer(FieldPath::from_name(VALUE_FIELD), Arc::new(indexed))
        .build()
}

/// Rows per block for [`pruning_test_batch`], and the number of blocks it spans.
const PRUNING_BLOCK_LEN: usize = 100;
const PRUNING_BLOCK_COUNT: usize = 5;
const PAYLOAD_FIELD: &str = "payload";

/// `value` is clustered by block: block `k` (rows `k * PRUNING_BLOCK_LEN` to
/// `(k + 1) * PRUNING_BLOCK_LEN - 1`) holds nothing but the constant `k + 1`. Filtering on a
/// single value therefore claims exactly one block as a match and leaves every other block fully
/// prunable.
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

async fn write_indexed_batch(
    store: &Arc<dyn ObjectStore>,
    session: &VortexSession,
    path: &Path,
    batch: &RecordBatch,
    block_len: usize,
    partition_len: Option<NonZeroU64>,
) -> anyhow::Result<()> {
    let array = session
        .arrow()
        .from_arrow_record_batch(batch.clone(), &batch.schema())?;
    let mut write = ObjectStoreWrite::new(Arc::clone(store), path).await?;
    session
        .write_options()
        .with_strategy(indexed_write_strategy(session, block_len, partition_len))
        .write(&mut write, array.to_array_stream())
        .await?;
    write.shutdown().await?;
    Ok(())
}

/// Total bytes physically read from storage, per [`InstrumentedReadAt`]'s
/// `vortex.io.read.total_size` counter.
fn total_bytes_read(registry: &DefaultMetricsRegistry) -> u64 {
    registry
        .snapshot()
        .into_iter()
        .filter(|metric| metric.name().as_ref() == "vortex.io.read.total_size")
        .filter_map(|metric| match metric.value() {
            MetricValue::Counter(counter) => Some(counter.value()),
            _ => None,
        })
        .sum()
}

/// Runs `value = <target>` against a freshly written copy of `batch`, projected down to
/// `payload`, and returns the matching rows plus the total bytes read from storage while
/// resolving them.
///
/// Writing to an in-memory [`ObjectStore`] and reading back through an
/// [`InstrumentedReadAt`]-wrapped [`ObjectStoreReadAt`] (rather than [`VortexFile::open_buffer`],
/// which [ignores metrics][open_buffer]) is what makes pruning observable: `object_store`'s byte-
/// range reads only pull in what the scan actually requests.
///
/// [`VortexFile::open_buffer`]: vortex_file::VortexFile
/// [open_buffer]: vortex_file::VortexOpenOptions::open_buffer
///
/// The footer is fetched once beforehand, outside the instrumented region: a real query engine
/// parses a file's footer once during planning and reuses it for every query that follows, so a
/// single query's pruning savings shouldn't be swamped by that one-time, query-independent cost —
/// especially for a file this small, where the footer can otherwise dominate the byte count.
async fn run_equality_filter(
    store: &Arc<dyn ObjectStore>,
    write_session: &VortexSession,
    read_session: VortexSession,
    batch: &RecordBatch,
    block_len: usize,
    partition_len: Option<NonZeroU64>,
    target: i32,
) -> anyhow::Result<(Vec<i32>, u64)> {
    let path = Path::from("files/indexed.vortex");
    write_indexed_batch(store, write_session, &path, batch, block_len, partition_len).await?;

    let footer = read_session
        .open_options()
        .open(Arc::new(ObjectStoreReadAt::new(
            Arc::clone(store),
            path.clone(),
            read_session.handle(),
        )))
        .await?
        .footer()
        .clone();

    let registry = Arc::new(DefaultMetricsRegistry::default());
    let reader = Arc::new(InstrumentedReadAt::new(
        Arc::new(ObjectStoreReadAt::new(
            Arc::clone(store),
            path,
            read_session.handle(),
        )),
        registry.as_ref(),
    ));
    let file = read_session
        .open_options()
        .with_footer(footer)
        .open(reader)
        .await?;

    let filter = eq(col(VALUE_FIELD), lit(target)).bind(file.dtype())?;
    let projection = select([PAYLOAD_FIELD], root()).bind(file.dtype())?;
    let result = file
        .scan()?
        .with_filter(filter)
        .with_projection(projection)
        .into_array_stream()?
        .read_all()
        .await?;

    let mut ctx = read_session.create_execution_ctx();
    let payload = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name(PAYLOAD_FIELD)?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    Ok((
        payload.as_slice::<i32>().to_vec(),
        total_bytes_read(&registry),
    ))
}

/// Correctness alone can't distinguish "the index answered the filter exactly" from "the index
/// was silently ignored and a full scan happened to get the right answer anyway" — both produce
/// identical query results. This test tells them apart by comparing bytes read from storage for
/// the same file and filter, once with [`ReverseIndex`] registered for reading and once without
/// (which forces the documented fallback: `IndexedReader::plan_probe` treats an unregistered
/// index kind's spec as inert). [`pruning_test_batch`] puts a single value in each block, so a
/// real exact-index probe lets the scan skip every block but one, while the unregistered run must
/// decode all of them to filter.
///
/// The partitioned case also round-trips the index's partitioning through the file footer.
#[rstest]
#[case::unpartitioned(None)]
#[case::partitioned(NonZeroU64::new(2 * PRUNING_BLOCK_LEN as u64))]
#[tokio::test]
async fn test_index_avoids_reading_pruned_blocks(
    #[case] partition_len: Option<NonZeroU64>,
) -> anyhow::Result<()> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let write_session = session_with_indexed_layout()?;
    let batch = pruning_test_batch()?;
    // Block 0, not a middle block: object_store's read coalescing merges nearby byte ranges into
    // one physical read, so a "hole" in the middle of the file still gets pulled in as padding. A
    // match confined to the first block is a genuine prefix, so skipping the rest of the file
    // actually shrinks the range requested.
    let target = 1;

    let (with_index_rows, with_index_bytes) = run_equality_filter(
        &store,
        &write_session,
        session_with_indexed_layout()?,
        &batch,
        PRUNING_BLOCK_LEN,
        partition_len,
        target,
    )
    .await?;
    let (without_index_rows, without_index_bytes) = run_equality_filter(
        &store,
        &write_session,
        session_without_reverse_index(),
        &batch,
        PRUNING_BLOCK_LEN,
        partition_len,
        target,
    )
    .await?;

    assert_eq!(with_index_rows, without_index_rows);
    assert_eq!(with_index_rows.len(), PRUNING_BLOCK_LEN);

    assert!(
        with_index_bytes < without_index_bytes,
        "expected the registered index to skip the prunable blocks and read fewer bytes than the \
         unregistered fallback, got {with_index_bytes} (indexed) vs {without_index_bytes} \
         (fallback)"
    );

    // The last block sits in a later partition, whose postings are partition-local: they must
    // still land on the right file rows.
    let last = i32::try_from(PRUNING_BLOCK_COUNT)?;
    let (rows, _) = run_equality_filter(
        &store,
        &write_session,
        session_with_indexed_layout()?,
        &batch,
        PRUNING_BLOCK_LEN,
        partition_len,
        last,
    )
    .await?;
    let first_row = (last - 1) * i32::try_from(PRUNING_BLOCK_LEN)?;
    assert_eq!(
        rows,
        (first_row..first_row + i32::try_from(PRUNING_BLOCK_LEN)?).collect::<Vec<_>>()
    );

    Ok(())
}
