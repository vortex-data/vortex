//! End-to-end tests for the generic wrapper.
//!
//! Concrete index kinds live elsewhere (see the `reverse_index` example under
//! `examples/reverse_index/`), so these exercise the machinery through a test-only
//! [`exact_value::ExactValueIndex`] instead.

use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::expr::eq;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::INDEXED_LAYOUT_ID;
use super::IndexConfig;
use super::IndexSessionExt;
use super::IndexedStrategy;
use crate::LayoutChildType;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::indexed::tests::exact_value::DecliningIndex;
use crate::layouts::indexed::tests::exact_value::ExactValueIndex;
use crate::layouts::indexed::tests::fixed_superset::FixedSupersetIndex;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::scan::scan_builder::ScanBuilder;
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

/// Writes the text column, returning the resulting layout and the segments backing it.
async fn write(
    session: &VortexSession,
    configs: Vec<IndexConfig>,
) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
    let ctx = ArrayContext::empty();
    let segments = Arc::new(TestSegments::default());
    let (ptr, eof) = SequenceId::root().split();

    let layout = strategy(configs)
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
) -> VortexResult<crate::LayoutReaderRef> {
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
        .ok_or_else(|| vortex_error::vortex_err!("an exact-value index was configured"))?;
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
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use crate::layouts::indexed::IndexBuilder;
    use crate::layouts::indexed::IndexExactness;
    use crate::layouts::indexed::IndexId;
    use crate::layouts::indexed::IndexQueryPlan;
    use crate::layouts::indexed::IndexResolve;
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

    #[derive(Debug)]
    pub struct ExactValueIndex;

    impl ExactValueIndex {
        pub fn new_ref() -> IndexVTableRef {
            Arc::new(Self)
        }
    }

    impl IndexVTable for ExactValueIndex {
        fn id(&self) -> IndexId {
            static ID: CachedId = CachedId::new(EXACT_VALUE_ID);
            *ID
        }

        fn supports_dtype(&self, dtype: &DType) -> bool {
            matches!(dtype, DType::Utf8(_))
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &[u8],
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<Box<dyn IndexBuilder>> {
            Ok(Box::new(Builder {
                postings: BTreeMap::new(),
            }))
        }

        fn plan(
            &self,
            expr: &BoundExpression,
            _dtype: &DType,
            _options: &[u8],
        ) -> VortexResult<Option<IndexQueryPlan>> {
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
                filter: eq(col(KEY_FIELD), lit(value.clone())),
                resolve: Arc::new(Resolve { value }),
            }))
        }
    }

    struct Builder {
        /// Sorted by construction, which is what gives the key column a useful zone map.
        postings: BTreeMap<String, RoaringBitmap>,
    }

    impl IndexBuilder for Builder {
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

        fn finish(self: Box<Self>) -> VortexResult<Option<(SendableArrayStream, Vec<u8>)>> {
            let mut keys = Vec::with_capacity(self.postings.len());
            let mut lists = Vec::with_capacity(self.postings.len());
            for (key, bitmap) in self.postings {
                let mut buffer = Vec::with_capacity(bitmap.serialized_size());
                bitmap
                    .serialize_into(&mut buffer)
                    .map_err(|err| vortex_err!("Failed to serialize postings: {err}"))?;
                keys.push(key);
                lists.push(buffer);
            }

            let len = keys.len();
            let array = StructArray::try_new_with_dtype(
                vec![
                    VarBinViewArray::from_iter_str(keys).into_array(),
                    VarBinViewArray::from_iter_bin(lists).into_array(),
                ],
                index_fields(),
                len,
                Validity::NonNullable,
            )?;

            Ok(Some((array.into_array().to_array_stream().boxed(), vec![])))
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
            Arc::new(Self)
        }
    }

    impl IndexVTable for DecliningIndex {
        fn id(&self) -> IndexId {
            static ID: CachedId = CachedId::new(DECLINING_ID);
            *ID
        }

        fn supports_dtype(&self, dtype: &DType) -> bool {
            matches!(dtype, DType::Utf8(_))
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &[u8],
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<Box<dyn IndexBuilder>> {
            Ok(Box::new(DecliningBuilder))
        }

        fn plan(
            &self,
            _expr: &BoundExpression,
            _dtype: &DType,
            _options: &[u8],
        ) -> VortexResult<Option<IndexQueryPlan>> {
            Ok(None)
        }
    }

    struct DecliningBuilder;

    impl IndexBuilder for DecliningBuilder {
        fn push(
            &mut self,
            _chunk: &ArrayRef,
            _row_offset: u64,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn finish(self: Box<Self>) -> VortexResult<Option<(SendableArrayStream, Vec<u8>)>> {
            Ok(None)
        }

        fn buffered_bytes(&self) -> u64 {
            0
        }
    }

    struct Resolve {
        value: String,
    }

    impl IndexResolve for Resolve {
        fn resolve(
            &self,
            postings: &ArrayRef,
            _data_row_count: u64,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<RowLocator> {
            let entries = postings.clone().execute::<StructArray>(ctx)?;
            let keys = entries
                .unmasked_field_by_name(KEY_FIELD)?
                .clone()
                .execute::<VarBinViewArray>(ctx)?;
            let lists = entries
                .unmasked_field_by_name(POSTINGS_FIELD)?
                .clone()
                .execute::<VarBinViewArray>(ctx)?;

            for idx in 0..keys.len() {
                if keys.bytes_at(idx).as_slice() != self.value.as_bytes() {
                    continue;
                }
                let bitmap = RoaringBitmap::deserialize_from(lists.bytes_at(idx).as_slice())
                    .map_err(|err| vortex_err!("Failed to deserialize postings: {err}"))?;
                return Ok(RowLocator::Rows(bitmap));
            }

            Ok(RowLocator::empty_rows())
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
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::layouts::indexed::IndexBuilder;
    use crate::layouts::indexed::IndexExactness;
    use crate::layouts::indexed::IndexId;
    use crate::layouts::indexed::IndexQueryPlan;
    use crate::layouts::indexed::IndexResolve;
    use crate::layouts::indexed::IndexVTable;
    use crate::layouts::indexed::IndexVTableRef;
    use crate::layouts::indexed::RowLocator;

    #[derive(Debug)]
    pub struct FixedSupersetIndex {
        id: &'static str,
        rows: RoaringBitmap,
    }

    impl FixedSupersetIndex {
        pub fn new_ref(id: &'static str, rows: RoaringBitmap) -> IndexVTableRef {
            Arc::new(Self { id, rows })
        }
    }

    impl IndexVTable for FixedSupersetIndex {
        fn id(&self) -> IndexId {
            IndexId::from(self.id)
        }

        fn supports_dtype(&self, dtype: &DType) -> bool {
            matches!(dtype, DType::Utf8(_))
        }

        fn builder(
            &self,
            _dtype: &DType,
            _options: &[u8],
            _data_block_len: Option<u64>,
            _session: &VortexSession,
        ) -> VortexResult<Box<dyn IndexBuilder>> {
            Ok(Box::new(Builder))
        }

        fn plan(
            &self,
            _expr: &BoundExpression,
            _dtype: &DType,
            _options: &[u8],
        ) -> VortexResult<Option<IndexQueryPlan>> {
            Ok(Some(IndexQueryPlan {
                exactness: IndexExactness::Superset,
                filter: eq(root(), lit(0i32)),
                resolve: Arc::new(Resolve {
                    rows: self.rows.clone(),
                }),
            }))
        }
    }

    /// Writes one dummy row so the layout has real content for `plan`'s filter to select; the
    /// value itself is never inspected.
    struct Builder;

    impl IndexBuilder for Builder {
        fn push(
            &mut self,
            _chunk: &ArrayRef,
            _row_offset: u64,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn finish(self: Box<Self>) -> VortexResult<Option<(SendableArrayStream, Vec<u8>)>> {
            let array = PrimitiveArray::from_iter([0i32]).into_array();
            Ok(Some((array.to_array_stream().boxed(), vec![])))
        }

        fn buffered_bytes(&self) -> u64 {
            0
        }
    }

    struct Resolve {
        rows: RoaringBitmap,
    }

    impl IndexResolve for Resolve {
        fn resolve(
            &self,
            _postings: &ArrayRef,
            _data_row_count: u64,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<RowLocator> {
            Ok(RowLocator::Rows(self.rows.clone()))
        }
    }
}
