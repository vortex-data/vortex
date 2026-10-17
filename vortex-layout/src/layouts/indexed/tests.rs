//! End-to-end tests for the generic wrapper.
//!
//! Concrete index kinds live elsewhere (see the `reverse_index` example under
//! `examples/reverse_index/`), so these exercise the machinery through a test-only
//! [`exact_value::ExactValueIndex`] instead.

use std::num::NonZeroU64;
use std::num::NonZeroUsize;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::channel::oneshot;
use parking_lot::Mutex;
use roaring::RoaringBitmap;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::eq;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::test_harness::check_metadata;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::INDEXED_LAYOUT_ID;
use super::IndexConfig;
use super::IndexPartitioning;
use super::IndexSpec;
use super::IndexSessionExt;
use super::Indexed;
use super::IndexedLayout;
use super::IndexedStrategy;
use super::test_harness::check_roundtrip;
use crate::LayoutChildType;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::Flat;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::indexed::tests::exact_value::DecliningIndex;
use crate::layouts::indexed::tests::exact_value::Entries;
use crate::layouts::indexed::tests::exact_value::ExactValueIndex;
use crate::layouts::indexed::tests::fixed_superset::FixedSupersetIndex;
use crate::layouts::indexed::tests::fixed_superset::INDEX_ROWS;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::scan_builder::ScanBuilder;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::test::new_session;

/// The only index kind these tests attach, standing in for a real plugin.
fn exact_configs() -> Vec<IndexConfig> {
    vec![IndexConfig::with_defaults(ExactValueIndex::new_ref())]
}

/// Small enough that a 12-row file spans three blocks, making the row/block granularity
/// difference visible in a single assertion.
const BLOCK_LEN: usize = 4;

/// Rows 1 and 9 contain "needle"; nothing else does. With `BLOCK_LEN` of 4 they land in
/// blocks 0 and 2, leaving block 1 prunable.
const ROWS: [&str; 12] = [
    "alpha",
    "a needle here",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "needle again",
    "iota",
    "kappa",
];

/// A session knowing the index kinds this test suite ships.
///
/// Deliberately not a shared global session: sessions clone by sharing one `Arc`, so registering
/// into a shared session would leak between tests, and
/// [`unregistered_index_kind_falls_back_to_the_data_child`] depends on two sessions with different
/// index registries.
fn session_with_exact_index() -> VortexSession {
    let session = new_session();
    session.indexes().register(ExactValueIndex::new_ref());
    session
}

fn text_column() -> ArrayRef {
    VarBinViewArray::from_iter_str(ROWS).into_array()
}

/// A write strategy that attaches `configs` to the text column.
///
/// The indexed wrapper sits directly above repartitioning — the same slot `ZonedStrategy` occupies
/// — so it sees whole chunks in row order and knows the data child's block size.
fn strategy(configs: Vec<IndexConfig>) -> IndexedStrategy {
    let data = RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: BLOCK_LEN,
            block_size_target: None,
            canonicalize: false,
        },
    );
    IndexedStrategy::new(data, FlatLayoutStrategy::default(), configs)
        .with_data_block_len(BLOCK_LEN as u64)
}

/// A write strategy whose data child is one 12-row chunk, far wider than the index partitions
/// the tests configure, with each index partition written as its own index-child chunk.
///
/// That is the shape partitioning exists for: index granularity chosen independently of how the
/// data is chunked.
fn partitioned_strategy(configs: Vec<IndexConfig>) -> IndexedStrategy {
    IndexedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        configs,
    )
    .with_data_block_len(BLOCK_LEN as u64)
}

fn partitioned(config: IndexConfig, partition_len: u64) -> VortexResult<IndexConfig> {
    let partition_len = NonZeroU64::new(partition_len)
        .ok_or_else(|| vortex_err!("partition length must be non-zero"))?;
    Ok(config.with_partition_len(partition_len))
}

/// Writes the text column, returning the resulting layout and the segments backing it.
async fn write(
    session: &VortexSession,
    configs: Vec<IndexConfig>,
) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
    write_with(session, strategy(configs)).await
}

async fn write_with(
    session: &VortexSession,
    strategy: IndexedStrategy,
) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
    let ctx = ArrayContext::empty();
    let segments = Arc::new(TestSegments::default());
    let (ptr, eof) = SequenceId::root().split();

    let layout = strategy
        .write_stream(
            ctx.into(),
            Arc::<TestSegments>::clone(&segments),
            text_column().to_array_stream().sequenced(ptr),
            eof,
            session,
        )
        .await?;
    Ok((layout, segments))
}

fn text_reader(
    session: &VortexSession,
    layout: &LayoutRef,
    segments: Arc<TestSegments>,
) -> VortexResult<LayoutReaderRef> {
    layout.new_reader("text".into(), segments, session, &Default::default())
}

/// The pruning mask the text reader produces for `LIKE '%needle%'`.
///
/// Goes straight at the indexed reader rather than through a scan, so the mask under test is
/// unambiguously the one the index produced.
async fn prune_mask(
    session: &VortexSession,
    layout: &LayoutRef,
    segments: Arc<TestSegments>,
    needle: &str,
) -> VortexResult<Mask> {
    let reader = text_reader(session, layout, segments)?;
    let row_count = reader.row_count();
    let filter = like(root(), lit(format!("%{needle}%"))).bind(reader.dtype())?;

    reader
        .pruning_evaluation(
            &(0..row_count),
            &filter,
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await
}

/// The mask the text reader produces for `text == value`.
///
/// An `Exact` plan serves `filter_evaluation` directly, so the result is the index's own answer
/// rather than the data child's, intersected with `input`.
async fn exact_mask(
    session: &VortexSession,
    layout: &LayoutRef,
    segments: Arc<TestSegments>,
    value: &str,
    input: MaskFuture,
) -> VortexResult<Mask> {
    let reader = text_reader(session, layout, segments)?;
    let row_count = reader.row_count();
    let filter = eq(root(), lit(value)).bind(reader.dtype())?;

    reader
        .filter_evaluation(&(0..row_count), &filter, input)?
        .await
}

async fn scan_matching(
    session: &VortexSession,
    layout: &LayoutRef,
    segments: Arc<TestSegments>,
    needle: &str,
) -> VortexResult<Vec<String>> {
    let reader = text_reader(session, layout, segments)?;
    let filter = like(root(), lit(format!("%{needle}%"))).bind(reader.dtype())?;

    let text = ScanBuilder::new(session.clone(), reader)
        .with_filter(filter)
        .into_array_stream()?
        .read_all()
        .await?;

    let mut ctx = session.create_execution_ctx();
    let text = text.execute::<VarBinViewArray>(&mut ctx)?;
    Ok((0..text.len())
        .map(|idx| String::from_utf8_lossy(text.bytes_at(idx).as_slice()).into_owned())
        .collect())
}

#[tokio::test]
async fn unregistered_index_kind_falls_back_to_the_data_child() -> VortexResult<()> {
    // Written by a session that knows the exact value index...
    let (layout, segments) = write(
        &session_with_exact_index(),
        vec![IndexConfig::with_defaults(ExactValueIndex::new_ref())],
    )
    .await?;

    // ...and read by one that does not. The spec goes inert, nothing probes the index child, and
    // the data child answers everything — indexes are strictly optional accelerators.
    let read_session = new_session();

    assert!(
        prune_mask(&read_session, &layout, Arc::clone(&segments), "needle")
            .await?
            .all_true()
    );
    assert_eq!(
        scan_matching(&read_session, &layout, segments, "needle").await?,
        vec![ROWS[1].to_string(), ROWS[9].to_string()],
    );
    Ok(())
}

#[tokio::test]
async fn exact_index_answers_the_filter_itself() -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, segments) = write(
        &session,
        vec![IndexConfig::with_defaults(ExactValueIndex::new_ref())],
    )
    .await?;

    let mask = exact_mask(
        &session,
        &layout,
        Arc::clone(&segments),
        ROWS[2],
        MaskFuture::new_true(ROWS.len()),
    )
    .await?;
    assert_eq!(mask, Mask::from_iter((0..ROWS.len()).map(|row| row == 2)));

    // The post-condition is that the result is intersected with the input mask, so an input that
    // excludes the match must yield nothing.
    let excluded = exact_mask(
        &session,
        &layout,
        segments,
        ROWS[2],
        MaskFuture::ready(Mask::from_iter((0..ROWS.len()).map(|row| row != 2))),
    )
    .await?;
    assert!(excluded.all_false());

    Ok(())
}

#[tokio::test]
async fn layout_carries_one_auxiliary_child_per_index() -> VortexResult<()> {
    let session = new_session();
    let (layout, _segments) = write(&session, exact_configs()).await?;

    assert_eq!(layout.encoding_id().as_str(), INDEXED_LAYOUT_ID);

    // The edge types are load-bearing beyond this crate: anything attributing segments to a role
    // walks for the nearest `Auxiliary` edge, so an index child hung off a transparent edge would
    // silently be counted as data.
    assert_eq!(
        (0..layout.nslots())
            .filter_map(|slot| layout.slot_type(slot))
            .collect::<Vec<_>>(),
        vec![
            LayoutChildType::Transparent("data".into()),
            LayoutChildType::Auxiliary("index:test.idx.exact_value".into()),
        ],
    );

    // Index content is an ordinary layout tree, so it inherits chunking and zone maps for free.
    let index_child = layout
        .slot(1)?
        .ok_or_else(|| vortex_err!("an exact-value index was configured"))?;
    assert_eq!(
        index_child
            .dtype()
            .as_struct_fields()
            .names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>(),
        vec!["key".to_string(), "postings".to_string()],
    );
    assert!(index_child.row_count() > 0);
    Ok(())
}

/// A builder that finds nothing worth keeping must leave no trace at all: no index child, no spec,
/// and no wrapper, so the file reads exactly as if no index had been configured.
#[tokio::test]
async fn every_builder_declining_writes_no_wrapper() -> VortexResult<()> {
    let session = new_session();
    let (layout, segments) = write(
        &session,
        vec![IndexConfig::with_defaults(DecliningIndex::new_ref())],
    )
    .await?;

    assert_ne!(layout.encoding_id().as_str(), INDEXED_LAYOUT_ID);
    assert_eq!(
        scan_matching(&session, &layout, segments, "needle").await?,
        vec![ROWS[1].to_string(), ROWS[9].to_string()],
    );
    Ok(())
}

/// One index declining must not disturb the others: the wrapper survives with exactly the children
/// that were actually built, and the declining kind leaves no spec behind to probe.
#[tokio::test]
async fn one_builder_declining_leaves_the_others_intact() -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, segments) = write(
        &session,
        vec![
            IndexConfig::with_defaults(DecliningIndex::new_ref()),
            IndexConfig::with_defaults(ExactValueIndex::new_ref()),
        ],
    )
    .await?;

    assert_eq!(layout.encoding_id().as_str(), INDEXED_LAYOUT_ID);
    assert_eq!(
        layout
            .child_names()
            .map(|name| name.to_string())
            .collect::<Vec<_>>(),
        vec!["data".to_string(), "index:test.idx.exact_value".to_string()],
    );

    // The surviving index still answers, so the spec ordering did not shift out from under it.
    let mask = exact_mask(
        &session,
        &layout,
        segments,
        ROWS[2],
        MaskFuture::new_true(ROWS.len()),
    )
    .await?;
    assert_eq!(mask, Mask::from_iter((0..ROWS.len()).map(|row| row == 2)));
    Ok(())
}

/// Two `Superset` claims on the same conjunct must combine, not pick one and drop the other:
/// neither `{1, 2, 9}` nor `{2, 9, 10}` alone leaves only `{2, 9}` standing, only their
/// intersection does.
#[tokio::test]
async fn multiple_superset_claims_intersect() -> VortexResult<()> {
    let session = new_session();
    let (layout, segments) = write(
        &session,
        vec![
            IndexConfig::with_defaults(FixedSupersetIndex::new_ref(
                "test.idx.fixed_a",
                RoaringBitmap::from_iter([1u32, 2, 9]),
            )),
            IndexConfig::with_defaults(FixedSupersetIndex::new_ref(
                "test.idx.fixed_b",
                RoaringBitmap::from_iter([2u32, 9, 10]),
            )),
        ],
    )
    .await?;

    let reader = text_reader(&session, &layout, segments)?;
    let row_count = reader.row_count();
    // `FixedSupersetIndex` claims unconditionally, so any bound Utf8 conjunct exercises it.
    let filter = eq(root(), lit("irrelevant")).bind(reader.dtype())?;
    let mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter,
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;

    assert_eq!(
        mask,
        Mask::from_iter((0..ROWS.len()).map(|row| row == 2 || row == 9))
    );
    Ok(())
}

/// An `Exact` claim must discard an earlier, misleading `Superset` claim on the same conjunct
/// rather than intersect with it: the empty superset here would prune away row 2 if it were kept
/// around, but the exact claim that follows it proves row 2 is the real, correct answer.
#[tokio::test]
async fn exact_claim_discards_a_preceding_superset_claim() -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, segments) = write(
        &session,
        vec![
            IndexConfig::with_defaults(FixedSupersetIndex::new_ref(
                "test.idx.fixed_empty",
                RoaringBitmap::new(),
            )),
            IndexConfig::with_defaults(ExactValueIndex::new_ref()),
        ],
    )
    .await?;

    let reader = text_reader(&session, &layout, segments)?;
    let row_count = reader.row_count();
    let filter = eq(root(), lit(ROWS[2])).bind(reader.dtype())?;
    let mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter,
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;

    assert_eq!(mask, Mask::from_iter((0..ROWS.len()).map(|row| row == 2)));
    Ok(())
}

fn i64_dtype() -> DType {
    DType::Primitive(PType::I64, Nullability::NonNullable)
}

/// The pruning mask for a Utf8 conjunct `FixedSupersetIndex` claims unconditionally.
async fn superset_prune_mask(reader: &LayoutReaderRef) -> VortexResult<Mask> {
    let row_count = reader.row_count();
    let filter = eq(root(), lit("irrelevant")).bind(reader.dtype())?;
    reader
        .pruning_evaluation(
            &(0..row_count),
            &filter,
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await
}

/// A kind's declared index dtype is its schema, so the writer must refuse a builder whose output
/// differs from it rather than record a spec readers would probe with the wrong filter.
#[tokio::test]
async fn writer_rejects_an_index_that_does_not_match_its_declared_dtype() -> VortexResult<()> {
    let session = new_session();
    let result = write(
        &session,
        vec![IndexConfig::with_defaults(FixedSupersetIndex::declaring(
            "test.idx.misdeclared",
            RoaringBitmap::from_iter([2u32]),
            i64_dtype(),
        ))],
    )
    .await;

    let Err(err) = result else {
        vortex_bail!("writing an index that does not match its declared dtype should fail");
    };
    assert!(err.to_string().contains("declared"), "{err}");
    Ok(())
}

/// An index whose stored dtype its kind no longer declares, say one written before the kind's
/// schema changed, is skipped rather than probed with a filter bound to the wrong schema.
#[tokio::test]
async fn index_whose_stored_dtype_no_longer_matches_its_kind_is_skipped() -> VortexResult<()> {
    let session = new_session();
    let id = "test.idx.fixed";
    let rows = RoaringBitmap::from_iter([2u32]);
    let (layout, segments) = write(
        &session,
        vec![IndexConfig::with_defaults(FixedSupersetIndex::new_ref(
            id,
            rows.clone(),
        ))],
    )
    .await?;

    // As written, the index prunes.
    let reader = text_reader(&session, &layout, Arc::clone(&segments))?;
    assert_eq!(
        superset_prune_mask(&reader).await?,
        Mask::from_iter((0..ROWS.len()).map(|row| row == 2))
    );

    // The same file, read by a kind that now declares a different schema.
    let spec = &layout.as_::<Indexed>().indexes()[0];
    let changed = IndexSpec::new(
        FixedSupersetIndex::declaring(id, rows, i64_dtype()),
        spec.options().to_vec(),
        spec.index_dtype().clone(),
        spec.partitioning().cloned(),
    );
    let data = layout
        .slot(0)?
        .ok_or_else(|| vortex_err!("indexed layout has a data child"))?;
    let index = layout
        .slot(1)?
        .ok_or_else(|| vortex_err!("an index was configured"))?;
    let changed = IndexedLayout::try_new(data, vec![index], vec![changed])?.into_layout();

    let reader = text_reader(&session, &changed, segments)?;
    assert!(superset_prune_mask(&reader).await?.all_true());
    Ok(())
}

/// A kind's serialized form is a contract with every file already written, so each test kind's
/// content and options must survive their own encoding.
#[rstest]
#[case::empty(vec![])]
#[case::one(vec![("alpha", RoaringBitmap::from_iter([0u32]))])]
#[case::several(vec![
    ("alpha", RoaringBitmap::from_iter([0u32, 7])),
    ("beta", RoaringBitmap::new()),
    ("gamma", RoaringBitmap::from_iter([3u32, 4, 5, 100_000])),
])]
fn exact_value_index_roundtrips(
    #[case] entries: Vec<(&'static str, RoaringBitmap)>,
) -> VortexResult<()> {
    let vtable = ExactValueIndex::new_ref();
    let kind = vtable
        .as_opt::<ExactValueIndex>()
        .ok_or_else(|| vortex_err!("new_ref makes an ExactValueIndex"))?;
    let mut ctx = new_session().create_execution_ctx();
    check_roundtrip(
        kind,
        &DType::Utf8(Nullability::Nullable),
        &Entries::new(entries),
        &(),
        &mut ctx,
    )
}

#[test]
fn fixed_superset_index_roundtrips() -> VortexResult<()> {
    let vtable = FixedSupersetIndex::new_ref("test.idx.fixed", RoaringBitmap::new());
    let kind = vtable
        .as_opt::<FixedSupersetIndex>()
        .ok_or_else(|| vortex_err!("new_ref makes a FixedSupersetIndex"))?;
    let mut ctx = new_session().create_execution_ctx();
    check_roundtrip(
        kind,
        &DType::Utf8(Nullability::NonNullable),
        &(),
        &(),
        &mut ctx,
    )
}

/// Pins the indexed layout's own metadata: one spec per index with its id, options, index dtype
/// and partitioning. Covers both an unpartitioned index and a partitioned one with a declined
/// partition, so any change to how specs serialize shows up as a diff against the checked-in file.
#[cfg_attr(miri, ignore)]
#[tokio::test]
async fn indexed_layout_metadata() -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, _segments) = write_with(
        &session,
        partitioned_strategy(vec![
            partitioned(
                IndexConfig::with_defaults(ExactValueIndex::declining_partitions_containing(
                    ROWS[5],
                )),
                BLOCK_LEN as u64,
            )?,
            IndexConfig::with_defaults(FixedSupersetIndex::new_ref(
                "test.idx.fixed",
                RoaringBitmap::from_iter([2u32]),
            )),
        ]),
    )
    .await?;

    check_metadata("indexed.metadata", &layout.metadata());
    Ok(())
}

/// Rows where `ROWS[row] == value`, over `row_range`.
fn expected_rows(row_range: &Range<u64>, value: &str) -> VortexResult<Mask> {
    let rows = usize::try_from(row_range.start)?..usize::try_from(row_range.end)?;
    Ok(Mask::from_iter(rows.map(|row| ROWS[row] == value)))
}

fn eq_filter(reader: &LayoutReaderRef, value: &str) -> VortexResult<BoundExpression> {
    eq(root(), lit(value)).bind(reader.dtype())
}

/// Index content and metadata for the first (only) index on `layout`.
fn partitioning(layout: &LayoutRef) -> VortexResult<IndexPartitioning> {
    layout.as_::<Indexed>().indexes()[0]
        .partitioning()
        .cloned()
        .ok_or_else(|| vortex_err!("index should be partitioned"))
}

/// Partitioning changes how the index is built and probed, never what it answers: every value
/// must resolve to exactly its rows over any range, including ranges that straddle partitions and
/// a final partition shorter than the rest.
#[rstest]
#[tokio::test]
async fn partitioned_index_answers_like_an_unpartitioned_one(
    #[values(4, 8)] partition_len: u64,
    #[values(0..12, 2..10, 5..6, 4..8, 9..12)] row_range: Range<u64>,
) -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, segments) = write_with(
        &session,
        partitioned_strategy(vec![partitioned(
            IndexConfig::with_defaults(ExactValueIndex::new_ref()),
            partition_len,
        )?]),
    )
    .await?;

    // One data chunk, several index partitions: the two widths really are independent.
    let data = layout
        .slot(0)?
        .ok_or_else(|| vortex_err!("indexed layout has a data child"))?;
    assert!(data.as_opt::<Flat>().is_some());
    assert_eq!(
        partitioning(&layout)?.index_ends().len() as u64,
        (ROWS.len() as u64).div_ceil(partition_len)
    );

    let reader = text_reader(&session, &layout, segments)?;
    let len = usize::try_from(row_range.end - row_range.start)?;
    for value in [ROWS[1], ROWS[2], ROWS[9], ROWS[11], "absent"] {
        let mask = reader
            .filter_evaluation(
                &row_range,
                &eq_filter(&reader, value)?,
                MaskFuture::new_true(len),
            )?
            .await?;
        assert_eq!(mask, expected_rows(&row_range, value)?, "value {value:?}");
    }
    Ok(())
}

/// Records every segment requested, so a test can see which index bytes a probe touched.
struct CountingSegments {
    inner: Arc<TestSegments>,
    requested: Mutex<Vec<SegmentId>>,
}

impl SegmentSource for CountingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requested.lock().push(id);
        self.inner.request(id)
    }
}

fn segment_ids(layout: &LayoutRef) -> VortexResult<Vec<SegmentId>> {
    let mut ids = layout.segment_ids();
    for slot in 0..layout.nslots() {
        if let Some(child) = layout.slot(slot)? {
            ids.extend(segment_ids(&child)?);
        }
    }
    Ok(ids)
}

/// A reader over a file partitioned one block per index-child chunk, recording segment requests.
struct CountingFixture {
    reader: LayoutReaderRef,
    counting: Arc<CountingSegments>,
    /// Segments of each index partition, by partition.
    partitions: Vec<Vec<SegmentId>>,
}

impl CountingFixture {
    async fn new() -> VortexResult<Self> {
        let session = session_with_exact_index();
        let (layout, segments) = write_with(
            &session,
            partitioned_strategy(vec![partitioned(
                IndexConfig::with_defaults(ExactValueIndex::new_ref()),
                BLOCK_LEN as u64,
            )?]),
        )
        .await?;

        let index = layout
            .slot(1)?
            .ok_or_else(|| vortex_err!("an index was configured"))?;
        // Each partition is its own index-child chunk, so chunk `p` holds partition `p`.
        assert_eq!(index.nslots(), 3);
        let partitions = (0..index.nslots())
            .map(|p| {
                segment_ids(
                    &index
                        .slot(p)?
                        .ok_or_else(|| vortex_err!("partition {p} has a chunk"))?,
                )
            })
            .collect::<VortexResult<_>>()?;

        let counting = Arc::new(CountingSegments {
            inner: segments,
            requested: Mutex::new(Vec::new()),
        });
        let reader = layout.new_reader(
            "text".into(),
            Arc::<CountingSegments>::clone(&counting),
            &session,
            &Default::default(),
        )?;
        Ok(Self {
            reader,
            counting,
            partitions,
        })
    }

    /// How many index segment requests have been made so far, counting repeats.
    fn index_requests(&self) -> usize {
        let all_index: Vec<_> = self.partitions.iter().flatten().copied().collect();
        self.counting
            .requested
            .lock()
            .iter()
            .filter(|id| all_index.contains(id))
            .count()
    }

    /// The index segments requested so far, sorted and deduplicated.
    fn index_requested(&self) -> Vec<SegmentId> {
        let all_index: Vec<_> = self.partitions.iter().flatten().copied().collect();
        let mut requested: Vec<_> = self
            .counting
            .requested
            .lock()
            .iter()
            .copied()
            .filter(|id| all_index.contains(id))
            .collect();
        requested.sort();
        requested.dedup();
        requested
    }
}

/// The point of partitioning: a split covering one partition probes only that partition's slice
/// of the index child, not the whole index.
#[tokio::test]
async fn probing_a_range_reads_only_its_partitions_index() -> VortexResult<()> {
    let fixture = CountingFixture::new().await?;
    let reader = &fixture.reader;

    let row_range = 4..8;
    let mask = reader
        .filter_evaluation(
            &row_range,
            &eq_filter(reader, ROWS[5])?,
            MaskFuture::new_true(4),
        )?
        .await?;
    assert_eq!(mask, expected_rows(&row_range, ROWS[5])?);
    assert_eq!(fixture.index_requested(), fixture.partitions[1]);
    Ok(())
}

/// An index chunk is decoded once per reader, not once per expression: a second value looked up in
/// the same partition is answered from the chunk the first lookup decoded, with no further IO.
#[tokio::test]
async fn expressions_probing_the_same_partition_share_its_decoded_index() -> VortexResult<()> {
    let fixture = CountingFixture::new().await?;
    let reader = &fixture.reader;
    let row_range = 4..8;

    let mut requests = Vec::new();
    for value in [ROWS[5], ROWS[6]] {
        let mask = reader
            .filter_evaluation(
                &row_range,
                &eq_filter(reader, value)?,
                MaskFuture::new_true(4),
            )?
            .await?;
        assert_eq!(mask, expected_rows(&row_range, value)?, "value {value:?}");
        requests.push(fixture.index_requests());
    }

    assert_eq!(fixture.index_requested(), fixture.partitions[1]);
    // The first lookup loaded partition 1's index; the second loaded nothing more.
    assert!(requests[0] > 0);
    assert_eq!(requests[1], requests[0]);
    Ok(())
}

/// Decoding chunks rather than scanning must not cost the index child's zone maps: a probe still
/// prunes the chunks its filter rules out, and loads only the one that can hold its key.
#[tokio::test]
async fn probing_loads_only_the_index_chunks_its_filter_cannot_prune() -> VortexResult<()> {
    let session = new_session();
    // Two index rows per chunk, with a zone per chunk.
    let index = RepartitionStrategy::new(
        ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(2).ok_or_else(|| vortex_err!("non-zero"))?,
                ..Default::default()
            },
        ),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 2,
            block_size_target: None,
            canonicalize: false,
        },
    );
    let rows = RoaringBitmap::from_iter([3u32]);
    let (layout, segments) = write_with(
        &session,
        IndexedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            index,
            vec![IndexConfig::with_defaults(FixedSupersetIndex::probing(
                "test.idx.probing",
                rows,
                7,
            ))],
        ),
    )
    .await?;

    let index = layout
        .slot(1)?
        .ok_or_else(|| vortex_err!("an index was configured"))?;
    let chunks = index
        .slot(0)?
        .ok_or_else(|| vortex_err!("the zoned index child has data"))?;
    let chunk_count = usize::try_from(INDEX_ROWS)? / 2;
    assert_eq!(chunks.nslots(), chunk_count);
    let chunk_segments = (0..chunk_count)
        .map(|chunk| {
            segment_ids(
                &chunks
                    .slot(chunk)?
                    .ok_or_else(|| vortex_err!("chunk {chunk} exists"))?,
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let counting = Arc::new(CountingSegments {
        inner: segments,
        requested: Mutex::new(Vec::new()),
    });
    let reader = layout.new_reader(
        "text".into(),
        Arc::<CountingSegments>::clone(&counting),
        &session,
        &Default::default(),
    )?;
    assert_eq!(
        superset_prune_mask(&reader).await?,
        Mask::from_iter((0..ROWS.len()).map(|row| row == 3))
    );

    let requested = counting.requested.lock().clone();
    let loaded: Vec<usize> = (0..chunk_count)
        .filter(|chunk| {
            chunk_segments[*chunk]
                .iter()
                .any(|id| requested.contains(id))
        })
        .collect();
    // Index row 7 sits in the chunk of rows 6..8.
    assert_eq!(loaded, vec![3]);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Evaluation {
    Pruning,
    FilterReady,
    /// The selection is still pending when the filter is planned, as it is behind real pruning.
    FilterPending,
}

/// A split wider than a partition still skips the index of any partition its selection rules out
/// entirely, whether the selection arrives resolved or has to be awaited first.
#[rstest]
#[tokio::test]
async fn probing_skips_partitions_the_selection_rules_out(
    #[values(
        Evaluation::Pruning,
        Evaluation::FilterReady,
        Evaluation::FilterPending
    )]
    evaluation: Evaluation,
) -> VortexResult<()> {
    let fixture = CountingFixture::new().await?;
    let reader = &fixture.reader;

    let all = 0..ROWS.len() as u64;
    let selection = Mask::from_iter((0..ROWS.len()).map(|row| (4..8).contains(&row)));
    let filter = eq_filter(reader, ROWS[5])?;

    let mask = match evaluation {
        Evaluation::Pruning => {
            reader
                .pruning_evaluation(&all, &filter, selection.clone())?
                .await?
        }
        Evaluation::FilterReady => {
            reader
                .filter_evaluation(&all, &filter, MaskFuture::ready(selection.clone()))?
                .await?
        }
        Evaluation::FilterPending => {
            let (send, recv) = oneshot::channel();
            let pending = MaskFuture::new(ROWS.len(), async move {
                recv.await
                    .map_err(|_| vortex_err!("selection sender dropped"))
            });
            let evaluated = reader.filter_evaluation(&all, &filter, pending)?;
            // Nothing can be skipped before the selection is known, so nothing is probed yet.
            assert!(fixture.index_requested().is_empty());
            send.send(selection.clone())
                .map_err(|_| vortex_err!("filter dropped the selection"))?;
            evaluated.await?
        }
    };

    assert_eq!(mask, expected_rows(&all, ROWS[5])?.bitand(&selection));
    assert_eq!(fixture.index_requested(), fixture.partitions[1]);
    Ok(())
}

/// A partition whose builder declined knows nothing about its rows. Pruning must leave them all
/// alive, and an exact claim must not answer a split overlapping it, so the data child filters
/// those rows instead — while the built partitions keep pruning and answering exactly.
#[tokio::test]
async fn declined_partition_prunes_nothing_and_defers_to_the_data_child() -> VortexResult<()> {
    let session = session_with_exact_index();
    let (layout, segments) = write_with(
        &session,
        partitioned_strategy(vec![partitioned(
            IndexConfig::with_defaults(ExactValueIndex::declining_partitions_containing(ROWS[5])),
            BLOCK_LEN as u64,
        )?]),
    )
    .await?;

    let partitioning = partitioning(&layout)?;
    assert_eq!(partitioning.declined(), &RoaringBitmap::from_iter([1u32]));
    // The declined partition wrote no index rows, so its span is empty.
    let ends = partitioning.index_ends();
    assert_eq!(ends[0], ends[1]);

    let reader = text_reader(&session, &layout, segments)?;
    let all = 0..ROWS.len() as u64;

    let pruned = reader
        .pruning_evaluation(
            &all,
            &eq_filter(&reader, ROWS[2])?,
            Mask::new_true(ROWS.len()),
        )?
        .await?;
    assert_eq!(
        pruned,
        Mask::from_iter((0..ROWS.len()).map(|row| row == 2 || (4..8).contains(&row)))
    );

    // The value only the declined partition holds is still found, through the data child.
    for value in [ROWS[5], ROWS[2]] {
        let mask = reader
            .filter_evaluation(
                &all,
                &eq_filter(&reader, value)?,
                MaskFuture::new_true(ROWS.len()),
            )?
            .await?;
        assert_eq!(mask, expected_rows(&all, value)?, "value {value:?}");
    }
    Ok(())
}

/// Partitions must never cut through a data block, so the writer rejects a partition length it
/// cannot check against the block length, or one that is not a multiple of it.
#[rstest]
#[case::no_data_block_len(None, 4)]
#[case::misaligned(Some(4), 6)]
#[tokio::test]
async fn partition_len_must_align_with_the_data_block_len(
    #[case] data_block_len: Option<u64>,
    #[case] partition_len: u64,
) -> VortexResult<()> {
    let session = session_with_exact_index();
    let configs = vec![partitioned(
        IndexConfig::with_defaults(ExactValueIndex::new_ref()),
        partition_len,
    )?];
    let mut strategy = IndexedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        configs,
    );
    if let Some(block_len) = data_block_len {
        strategy = strategy.with_data_block_len(block_len);
    }

    assert!(write_with(&session, strategy).await.is_err());
    Ok(())
}

/// A test-only sorted value index, present to exercise the [`super::IndexExactness::Exact`] path
/// that a real posting-list index kind (such as an n-gram index) would rarely reach for equality
/// queries.
///
/// One row per distinct string, sorted, with a roaring posting list of the rows holding it. That
/// makes equality answerable outright, so `filter_evaluation` returns the index's mask and the
/// data child is never decoded for that conjunct.
mod exact_value {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use roaring::RoaringBitmap;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::arrays::varbinview::VarBinViewArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::literal::Literal;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use crate::layouts::indexed::IndexBuilder;
    use crate::layouts::indexed::IndexExactness;
    use crate::layouts::indexed::IndexId;
    use crate::layouts::indexed::IndexQueryPlan;
    use crate::layouts::indexed::IndexVTable;
    use crate::layouts::indexed::IndexVTableRef;
    use crate::layouts::indexed::RowLocator;

    pub const EXACT_VALUE_ID: &str = "test.idx.exact_value";
    pub const DECLINING_ID: &str = "test.idx.declining";
    const KEY_FIELD: &str = "key";
    const POSTINGS_FIELD: &str = "postings";

    fn index_fields() -> StructFields {
        let names: FieldNames = vec![KEY_FIELD, POSTINGS_FIELD].into();
        StructFields::new(
            names,
            vec![DType::Utf8(NonNullable), DType::Binary(NonNullable)],
        )
    }

    fn index_dtype() -> DType {
        DType::Struct(index_fields(), NonNullable)
    }

    #[derive(Debug)]
    pub struct ExactValueIndex {
        /// Write-side only: decline any partition holding this value. Reading is unaffected, so a
        /// session with the plain kind registered reads files written with this one.
        decline_if_contains: Option<&'static str>,
    }

    impl ExactValueIndex {
        pub fn new_ref() -> IndexVTableRef {
            IndexVTableRef::new(Self {
                decline_if_contains: None,
            })
        }

        pub fn declining_partitions_containing(value: &'static str) -> IndexVTableRef {
            IndexVTableRef::new(Self {
                decline_if_contains: Some(value),
            })
        }
    }

    /// A run of index entries: sorted, distinct keys and the rows holding each.
    #[derive(Debug)]
    pub struct Entries {
        keys: Vec<String>,
        postings: Vec<RoaringBitmap>,
    }

    impl Entries {
        pub fn new(entries: impl IntoIterator<Item = (&'static str, RoaringBitmap)>) -> Self {
            let (keys, postings) = entries
                .into_iter()
                .map(|(key, rows)| (key.to_string(), rows))
                .unzip();
            Self { keys, postings }
        }
    }

    impl IndexVTable for ExactValueIndex {
        type Options = ();
        type Builder = Builder;
        /// The value whose rows are wanted.
        type Query = String;
        type Chunk = Entries;

        fn id(&self) -> IndexId {
            static ID: CachedId = CachedId::new(EXACT_VALUE_ID);
            *ID
        }

        fn serialize_options(&self, _options: &()) -> Vec<u8> {
            vec![]
        }

        fn deserialize_options(&self, _options: &[u8]) -> VortexResult<()> {
            Ok(())
        }

        fn index_dtype(&self, dtype: &DType, _options: &()) -> Option<DType> {
            matches!(dtype, DType::Utf8(_)).then(index_dtype)
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &(),
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<Builder> {
            Ok(Builder {
                postings: BTreeMap::new(),
                decline_if_contains: self.decline_if_contains,
            })
        }

        fn plan(
            &self,
            expr: &BoundExpression,
            _dtype: &DType,
            index_dtype: &DType,
            _options: &(),
        ) -> VortexResult<Option<IndexQueryPlan<String>>> {
            // Only `<column> == <utf8 literal>`.
            if !expr.is::<Binary>() || *expr.as_::<Binary>() != Operator::Eq {
                return Ok(None);
            }
            if !expr.child(0).is_root() || !expr.child(1).is::<Literal>() {
                return Ok(None);
            }
            let Some(value) = expr.child(1).as_::<Literal>().as_utf8().value() else {
                return Ok(None);
            };
            let value = value.to_string();

            Ok(Some(IndexQueryPlan {
                exactness: IndexExactness::Exact,
                filter: eq(col(KEY_FIELD), lit(value.clone())).bind(index_dtype)?,
                query: value,
            }))
        }

        fn encode(
            &self,
            chunk: &Entries,
            _options: &(),
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            let mut lists = Vec::with_capacity(chunk.postings.len());
            for bitmap in &chunk.postings {
                let mut buffer = Vec::with_capacity(bitmap.serialized_size());
                bitmap
                    .serialize_into(&mut buffer)
                    .map_err(|err| vortex_err!("Failed to serialize postings: {err}"))?;
                lists.push(buffer);
            }

            Ok(StructArray::try_new_with_dtype(
                vec![
                    VarBinViewArray::from_iter_str(&chunk.keys).into_array(),
                    VarBinViewArray::from_iter_bin(lists).into_array(),
                ],
                index_fields(),
                chunk.keys.len(),
                Validity::NonNullable,
            )?
            .into_array())
        }

        fn decode(
            &self,
            chunk: ArrayRef,
            _options: &(),
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Entries> {
            let entries = chunk.execute::<StructArray>(ctx)?;
            let keys = entries
                .unmasked_field_by_name(KEY_FIELD)?
                .clone()
                .execute::<VarBinViewArray>(ctx)?;
            let lists = entries
                .unmasked_field_by_name(POSTINGS_FIELD)?
                .clone()
                .execute::<VarBinViewArray>(ctx)?;

            let mut decoded = Entries {
                keys: Vec::with_capacity(keys.len()),
                postings: Vec::with_capacity(keys.len()),
            };
            for idx in 0..keys.len() {
                decoded
                    .keys
                    .push(String::from_utf8_lossy(keys.bytes_at(idx).as_slice()).into_owned());
                decoded.postings.push(
                    RoaringBitmap::deserialize_from(lists.bytes_at(idx).as_slice())
                        .map_err(|err| vortex_err!("Failed to deserialize postings: {err}"))?,
                );
            }
            Ok(decoded)
        }

        fn resolve(
            &self,
            query: &String,
            chunks: &[Arc<Entries>],
            _data_row_count: u64,
            _options: &(),
        ) -> VortexResult<RowLocator> {
            for chunk in chunks {
                if let Ok(idx) = chunk.keys.binary_search(query) {
                    return Ok(RowLocator::Rows(chunk.postings[idx].clone()));
                }
            }
            Ok(RowLocator::empty_rows())
        }
    }

    pub struct Builder {
        /// Sorted by construction, which is what gives the key column a useful zone map.
        postings: BTreeMap<String, RoaringBitmap>,
        decline_if_contains: Option<&'static str>,
    }

    impl IndexBuilder for Builder {
        type Options = ();
        type Chunk = Entries;

        fn push(
            &mut self,
            chunk: &ArrayRef,
            row_offset: u64,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            let values = chunk.clone().execute::<VarBinViewArray>(ctx)?;
            let validity = values
                .varbinview_validity()
                .execute_mask(values.len(), ctx)?;

            for idx in 0..values.len() {
                if !validity.value(idx) {
                    continue;
                }
                let value = String::from_utf8_lossy(values.bytes_at(idx).as_slice()).into_owned();
                self.postings
                    .entry(value)
                    .or_default()
                    .insert(u32::try_from(row_offset + idx as u64)?);
            }
            Ok(())
        }

        fn finish(self) -> VortexResult<Option<(Vec<Entries>, ())>> {
            if self
                .decline_if_contains
                .is_some_and(|value| self.postings.contains_key(value))
            {
                return Ok(None);
            }
            let (keys, postings) = self.postings.into_iter().unzip();
            Ok(Some((vec![Entries { keys, postings }], ())))
        }

        fn buffered_bytes(&self) -> u64 {
            self.postings
                .values()
                .map(|bitmap| bitmap.serialized_size() as u64)
                .sum()
        }
    }

    /// A kind that always declines at `finish`.
    ///
    /// Standing in for "the index would not be worth its bytes", so the decline paths are testable
    /// without a fixture large enough to trip a real threshold.
    #[derive(Debug)]
    pub struct DecliningIndex;

    impl DecliningIndex {
        pub fn new_ref() -> IndexVTableRef {
            IndexVTableRef::new(Self)
        }
    }

    impl IndexVTable for DecliningIndex {
        type Options = ();
        type Builder = DecliningBuilder;
        type Query = ();
        type Chunk = ();

        fn id(&self) -> IndexId {
            static ID: CachedId = CachedId::new(DECLINING_ID);
            *ID
        }

        fn serialize_options(&self, _options: &()) -> Vec<u8> {
            vec![]
        }

        fn deserialize_options(&self, _options: &[u8]) -> VortexResult<()> {
            Ok(())
        }

        fn index_dtype(&self, dtype: &DType, _options: &()) -> Option<DType> {
            matches!(dtype, DType::Utf8(_)).then(index_dtype)
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &(),
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<DecliningBuilder> {
            Ok(DecliningBuilder)
        }

        fn plan(
            &self,
            _expr: &BoundExpression,
            _dtype: &DType,
            _index_dtype: &DType,
            _options: &(),
        ) -> VortexResult<Option<IndexQueryPlan<()>>> {
            Ok(None)
        }

        fn encode(&self, _chunk: &(), _options: &(), _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
            vortex_bail!("a declining index never builds content to encode")
        }

        fn decode(&self, _chunk: ArrayRef, _options: &(), _ctx: &mut ExecutionCtx) -> VortexResult<()> {
            Ok(())
        }

        fn resolve(
            &self,
            _query: &(),
            _chunks: &[Arc<()>],
            _data_row_count: u64,
            _options: &(),
        ) -> VortexResult<RowLocator> {
            Ok(RowLocator::empty_rows())
        }
    }

    pub struct DecliningBuilder;

    impl IndexBuilder for DecliningBuilder {
        type Options = ();
        type Chunk = ();

        fn push(
            &mut self,
            _chunk: &ArrayRef,
            _row_offset: u64,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn finish(self) -> VortexResult<Option<(Vec<()>, ())>> {
            Ok(None)
        }

        fn buffered_bytes(&self) -> u64 {
            0
        }
    }
}

/// A test-only index kind that always claims any expression with `Superset` exactness and answers
/// with a fixed locator supplied at construction, ignoring both the expression and the data.
///
/// Real superset-only kinds (bloom filters, n-gram indexes) derive their locator from the data;
/// this one is a stand-in that hands a test exact, known masks to combine, so its assertions are
/// about the reader's sibling-combination logic rather than any kind's own indexing correctness.
mod fixed_superset {
    use std::sync::Arc;

    use roaring::RoaringBitmap;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::layouts::indexed::IndexBuilder;
    use crate::layouts::indexed::IndexExactness;
    use crate::layouts::indexed::IndexId;
    use crate::layouts::indexed::IndexQueryPlan;
    use crate::layouts::indexed::IndexVTable;
    use crate::layouts::indexed::IndexVTableRef;
    use crate::layouts::indexed::RowLocator;

    /// Rows of the index child, holding `0..INDEX_ROWS` as `i32`s.
    pub const INDEX_ROWS: i32 = 12;

    #[derive(Debug)]
    pub struct FixedSupersetIndex {
        id: &'static str,
        rows: RoaringBitmap,
        /// What [`IndexVTable::index_dtype`] claims; the builder always writes `i32`s regardless.
        declared: DType,
        /// The index row value the plan's filter selects, so a zone-mapped index child can prune
        /// every chunk but the one holding it.
        probe: i32,
    }

    impl FixedSupersetIndex {
        pub fn new_ref(id: &'static str, rows: RoaringBitmap) -> IndexVTableRef {
            Self::declaring(id, rows, DType::Primitive(PType::I32, Nullability::NonNullable))
        }

        /// A kind that declares `declared` as its index dtype, standing in for a builder that
        /// emits something other than it declared, or a newer version of a kind whose schema has
        /// changed since a file was written.
        pub fn declaring(id: &'static str, rows: RoaringBitmap, declared: DType) -> IndexVTableRef {
            IndexVTableRef::new(Self {
                id,
                rows,
                declared,
                probe: 0,
            })
        }

        /// A kind whose plans select only the index row holding `probe`.
        pub fn probing(id: &'static str, rows: RoaringBitmap, probe: i32) -> IndexVTableRef {
            IndexVTableRef::new(Self {
                id,
                rows,
                declared: DType::Primitive(PType::I32, Nullability::NonNullable),
                probe,
            })
        }
    }

    impl IndexVTable for FixedSupersetIndex {
        type Options = ();
        type Builder = Builder;
        /// The fixed rows to answer with.
        type Query = RoaringBitmap;
        type Chunk = ();

        fn id(&self) -> IndexId {
            IndexId::from(self.id)
        }

        fn serialize_options(&self, _options: &()) -> Vec<u8> {
            vec![]
        }

        fn deserialize_options(&self, _options: &[u8]) -> VortexResult<()> {
            Ok(())
        }

        fn index_dtype(&self, dtype: &DType, _options: &()) -> Option<DType> {
            matches!(dtype, DType::Utf8(_)).then(|| self.declared.clone())
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &(),
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<Builder> {
            Ok(Builder)
        }

        fn plan(
            &self,
            _expr: &BoundExpression,
            _dtype: &DType,
            index_dtype: &DType,
            _options: &(),
        ) -> VortexResult<Option<IndexQueryPlan<RoaringBitmap>>> {
            Ok(Some(IndexQueryPlan {
                exactness: IndexExactness::Superset,
                filter: eq(root(), lit(self.probe)).bind(index_dtype)?,
                query: self.rows.clone(),
            }))
        }

        fn encode(&self, _chunk: &(), _options: &(), _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
            Ok(PrimitiveArray::from_iter(0..INDEX_ROWS).into_array())
        }

        fn decode(&self, _chunk: ArrayRef, _options: &(), _ctx: &mut ExecutionCtx) -> VortexResult<()> {
            Ok(())
        }

        fn resolve(
            &self,
            query: &RoaringBitmap,
            _chunks: &[Arc<()>],
            _data_row_count: u64,
            _options: &(),
        ) -> VortexResult<RowLocator> {
            Ok(RowLocator::Rows(query.clone()))
        }
    }

    /// Builds one chunk that encodes as `0..INDEX_ROWS`, so the layout has real content for `plan`'s
    /// filter to select; the values themselves are never decoded.
    pub struct Builder;

    impl IndexBuilder for Builder {
        type Options = ();
        type Chunk = ();

        fn push(
            &mut self,
            _chunk: &ArrayRef,
            _row_offset: u64,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn finish(self) -> VortexResult<Option<(Vec<()>, ())>> {
            Ok(Some((vec![()], ())))
        }

        fn buffered_bytes(&self) -> u64 {
            0
        }
    }
}
