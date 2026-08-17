// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A layout whose subtree is stored in a segment rather than inline in its parent's flatbuffer.
//!
//! A Vortex layout is a single recursive flatbuffer, parsed whole before any data is read, and
//! verified against a table limit. A paged layout breaks that recursion: it reports no inline
//! children and instead holds one segment containing its subtree as a nested `Layout` flatbuffer.
//! Each page is therefore verified independently, so the table and depth limits apply per page
//! rather than to the whole file.
//!
//! Because the segment fetch is asynchronous but [`LayoutChildren`](crate::LayoutChildren) is not,
//! the descent happens in the reader, whose evaluation methods already return futures.

mod reader;
pub mod writer;

use std::sync::Arc;

use itertools::Either;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::paged::reader::PagedReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Paged layout vtable.
#[derive(Clone, Debug)]
pub struct Paged;

/// Backwards-compatible name for the paged layout plugin.
pub use Paged as PagedLayoutEncoding;

/// Where a page's subtree splits into chunks, as recorded at its boundary.
///
/// Scan planning is synchronous and must not read the page, so these boundaries travel with it.
/// Storing one integer per chunk would make them the dominant cost of the footer at any useful
/// page size, so the uniform case — which is what a repartitioner emitting fixed-size row blocks
/// produces — is kept symbolic.
#[derive(Clone, Debug)]
pub enum ChunkBoundaries {
    /// Every chunk holds `len` rows, except possibly a shorter last one.
    Uniform {
        /// Rows per chunk.
        len: u64,
        /// Number of chunks.
        count: usize,
        /// Total rows, which the last chunk is truncated to.
        row_count: u64,
    },
    /// Exclusive row boundaries, for subtrees whose chunks differ in length.
    Explicit(Arc<[u64]>),
}

impl ChunkBoundaries {
    /// Choose a representation for `offsets`, the exclusive row boundaries of a subtree.
    pub fn from_offsets(offsets: &[u64], row_count: u64) -> Self {
        let uniform = offsets.split_last().is_some_and(|(last, rest)| {
            let len = offsets[0];
            *last == row_count
                && len > 0
                && rest
                    .iter()
                    .enumerate()
                    .all(|(idx, offset)| *offset == (idx as u64 + 1) * len)
                && row_count > (offsets.len() as u64 - 1) * len
                && row_count <= offsets.len() as u64 * len
        });

        if uniform {
            Self::Uniform {
                len: offsets[0],
                count: offsets.len(),
                row_count,
            }
        } else {
            Self::Explicit(offsets.into())
        }
    }

    /// The exclusive row boundaries, relative to the subtree's first row.
    pub fn offsets(&self) -> impl Iterator<Item = u64> + '_ {
        match self {
            Self::Uniform {
                len,
                count,
                row_count,
            } => Either::Left((1..=*count).map(move |idx| (idx as u64 * len).min(*row_count))),
            Self::Explicit(offsets) => Either::Right(offsets.iter().copied()),
        }
    }

    /// The number of chunks.
    pub fn len(&self) -> usize {
        match self {
            Self::Uniform { count, .. } => *count,
            Self::Explicit(offsets) => offsets.len(),
        }
    }

    /// Returns `true` if the subtree has no chunks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Paged-layout-specific data.
#[derive(Clone, Debug)]
pub struct PagedData {
    segment_id: SegmentId,
    layout_ctx: ReadContext,
    array_ctx: ReadContext,
    boundaries: ChunkBoundaries,
}

/// A layout standing in for a subtree serialized into its own segment.
pub type PagedLayout = Layout<Paged>;

impl VTable for Paged {
    type LayoutData = PagedData;
    type Metadata = ProstMetadata<PagedLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.paged");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        // No encoding dictionary: the page's flatbuffer is interned into the same context the
        // footer serializes through, so it indexes into the file's one dictionary.
        ProstMetadata(PagedLayoutMetadata {
            uniform_chunk_len: match &layout.boundaries {
                ChunkBoundaries::Uniform { len, .. } => Some(*len),
                ChunkBoundaries::Explicit(_) => None,
            },
            chunk_count: match &layout.boundaries {
                ChunkBoundaries::Uniform { count, .. } => Some(*count as u64),
                ChunkBoundaries::Explicit(_) => None,
            },
            row_offsets: match &layout.boundaries {
                ChunkBoundaries::Uniform { .. } => Vec::new(),
                ChunkBoundaries::Explicit(offsets) => offsets.to_vec(),
            },
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &PagedLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        if args.segment_ids.len() != 1 {
            vortex_bail!("Paged layout must have exactly one segment ID");
        }
        if args.children.nchildren() != 0 {
            vortex_bail!("Paged layout must not have inline children");
        }
        let boundaries = match (metadata.uniform_chunk_len, metadata.chunk_count) {
            (Some(len), Some(count)) => {
                let count = usize::try_from(count)?;
                if len == 0 || count == 0 {
                    vortex_bail!("Paged layout uniform chunk length and count must be non-zero");
                }
                // The last chunk may be short, but the rest must exactly cover the rows before it.
                if args.row_count <= (count as u64 - 1) * len || args.row_count > count as u64 * len
                {
                    vortex_bail!(
                        "Paged layout {count} chunks of {len} rows do not cover {} rows",
                        args.row_count
                    );
                }
                ChunkBoundaries::Uniform {
                    len,
                    count,
                    row_count: args.row_count,
                }
            }
            (None, None) => {
                if metadata
                    .row_offsets
                    .last()
                    .is_some_and(|last| *last != args.row_count)
                {
                    vortex_bail!("Paged layout row offsets do not add up to its row count");
                }
                ChunkBoundaries::Explicit(metadata.row_offsets.as_slice().into())
            }
            _ => vortex_bail!(
                "Paged layout must set both a uniform chunk length and a chunk count, or neither"
            ),
        };
        Ok(PagedData {
            segment_id: args.segment_ids[0],
            layout_ctx: args.layout_read_ctx.clone(),
            array_ctx: args.array_read_ctx.clone(),
            boundaries,
        })
    }

    fn child_dtype(_layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        vortex_bail!("Paged layout has no inline child {idx}; its subtree is in its segment")
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        vortex_panic!("Paged layout has no inline child {idx}; its subtree is in its segment")
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(PagedReader::new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )))
    }
}

impl Layout<Paged> {
    /// Construct a paged layout over an already-written page segment.
    ///
    /// `boundaries` are the subtree's chunk boundaries, used to plan scans without reading the
    /// page. They must cover exactly `row_count` rows.
    pub fn new(
        row_count: u64,
        dtype: DType,
        segment_id: SegmentId,
        layout_ctx: ReadContext,
        array_ctx: ReadContext,
        boundaries: ChunkBoundaries,
    ) -> Self {
        LayoutParts::new(
            Paged,
            dtype,
            row_count,
            vec![segment_id],
            OwnedLayoutChildren::layout_children(Vec::new()),
            PagedData {
                segment_id,
                layout_ctx,
                array_ctx,
                boundaries,
            },
        )
        .into_typed()
    }

    /// Returns the segment holding this page's subtree.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the layout encoding dictionary the page's flatbuffer is written against.
    pub fn layout_ctx(&self) -> &ReadContext {
        &self.layout_ctx
    }

    /// Returns the array read context used to decode arrays within the page.
    pub fn array_ctx(&self) -> &ReadContext {
        &self.array_ctx
    }

    /// Returns the subtree's chunk boundaries, relative to this page's first row.
    pub fn boundaries(&self) -> &ChunkBoundaries {
        &self.boundaries
    }
}

#[derive(prost::Message)]
pub struct PagedLayoutMetadata {
    /// Exclusive row boundaries of the subtree's chunks, relative to this page's first row.
    ///
    /// Packed varints, so promoting them to the page boundary costs no flatbuffer tables — the
    /// tables are what the verifier limits. Empty when the boundaries are uniform, since one
    /// integer per chunk would otherwise dominate the footer at any useful page size.
    #[prost(repeated, uint64, tag = "2")]
    pub row_offsets: Vec<u64>,
    /// Rows per chunk, when every chunk but the last holds the same number.
    #[prost(optional, uint64, tag = "3")]
    pub uniform_chunk_len: Option<u64>,
    /// Number of chunks, set together with [`Self::uniform_chunk_len`].
    #[prost(optional, uint64, tag = "4")]
    pub chunk_count: Option<u64>,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use flatbuffers::VerifierOptions;
    use flatbuffers::root_with_opts;
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_flatbuffers::WriteFlatBufferExt;
    use vortex_flatbuffers::layout as fbl;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

    use super::*;
    use crate::LayoutContext;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::LayoutWriterContext;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::scan::split_by::SplitBy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::test::new_session;

    /// Write `chunk_count` single-row chunks, optionally grouped into pages.
    async fn write_chunks(
        chunk_count: usize,
        page_size: usize,
        session: &VortexSession,
    ) -> (Arc<TestSegments>, LayoutRef) {
        write_chunk_lens(&vec![1i32; chunk_count], page_size, session).await
    }

    /// Write one chunk per entry in `chunk_lens`, optionally grouped into pages.
    async fn write_chunk_lens(
        chunk_lens: &[i32],
        page_size: usize,
        session: &VortexSession,
    ) -> (Arc<TestSegments>, LayoutRef) {
        let segments = Arc::new(TestSegments::default());
        let (mut sequence_id, eof) = SequenceId::root().split();

        let mut next = 0i32;
        let chunks = chunk_lens
            .iter()
            .map(|len| {
                let chunk: PrimitiveArray = (next..next + *len).collect();
                next += *len;
                Ok((sequence_id.advance(), chunk.into_array()))
            })
            .collect::<Vec<_>>();

        let layout = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
            .with_page_size(page_size)
            .write_stream(
                LayoutWriterContext::new(ArrayContext::empty()),
                Arc::<TestSegments>::clone(&segments),
                SequentialStreamAdapter::new(
                    DType::Primitive(PType::I32, NonNullable),
                    stream::iter(chunks),
                )
                .sendable(),
                eof,
                session,
            )
            .await
            .unwrap();

        (segments, layout)
    }

    fn to_flatbuffer(layout: &LayoutRef) -> Vec<u8> {
        layout
            .flatbuffer_writer(&LayoutContext::default())
            .write_flatbuffer_bytes()
            .unwrap()
            .to_vec()
    }

    fn verifies_within(bytes: &[u8], max_tables: usize) -> bool {
        let opts = VerifierOptions {
            max_tables,
            ..Default::default()
        };
        root_with_opts::<fbl::Layout>(&opts, bytes).is_ok()
    }

    /// A [`SegmentSource`] that counts requests, so IO can be asserted about.
    struct CountingSource {
        inner: Arc<dyn SegmentSource>,
        requests: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSource {
        fn request(&self, id: SegmentId) -> crate::segments::SegmentFuture {
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.inner.request(id)
        }
    }

    fn splits_of(
        session: &VortexSession,
        segments: Arc<TestSegments>,
        layout: &LayoutRef,
    ) -> Vec<u64> {
        let reader = layout
            .new_reader("".into(), segments, session, &Default::default())
            .unwrap();
        SplitBy::Layout
            .splits(reader.as_ref(), &(0..layout.row_count()), &[FieldMask::All])
            .unwrap()
    }

    /// Fixed-size row blocks are what the repartitioner emits, so a page must not spend a varint
    /// per chunk recording boundaries it could derive from two integers. Otherwise the offsets
    /// become the dominant cost of the footer at any useful page size.
    #[test]
    fn uniform_pages_record_boundaries_in_constant_space() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);

            // One page in each case, holding 8 and 64 equal-length chunks respectively.
            let (_, eight) = write_chunks(8, 8, &session).await;
            let (_, sixty_four) = write_chunks(64, 64, &session).await;

            let eight = eight.slot(0).unwrap().unwrap().metadata().len();
            let sixty_four = sixty_four.slot(0).unwrap().unwrap().metadata().len();

            assert!(
                sixty_four <= eight + 1,
                "a page of 64 uniform chunks should cost no more metadata than one of 8, \
                 but it took {sixty_four} bytes against {eight}"
            );
        })
    }

    /// A page must not repeat the file's layout encoding dictionary. Interning into the same
    /// context the footer uses means a page's metadata carries only its chunk boundaries, which
    /// for a uniform subtree is two integers.
    #[test]
    fn pages_do_not_repeat_the_encoding_dictionary() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let (_, paged) = write_chunks(64, 64, &session).await;

            let page = paged.slot(0).unwrap().unwrap();
            let metadata = page.metadata().len();
            assert!(
                metadata <= 16,
                "a uniform page should carry only its boundaries, but its metadata is \
                 {metadata} bytes"
            );
        })
    }

    /// Chunks of differing lengths cannot be derived, so those pages fall back to explicit
    /// offsets. The split set must stay exact either way.
    #[test]
    fn non_uniform_pages_keep_exact_splits() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let lens = [3i32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];

            let (inline_segments, inline) = write_chunk_lens(&lens, 0, &session).await;
            let (paged_segments, paged) = write_chunk_lens(&lens, 3, &session).await;

            assert_eq!(
                splits_of(&session, paged_segments, &paged),
                splits_of(&session, inline_segments, &inline)
            );
        })
    }

    /// Paging must not coarsen the scan's batch boundaries. The page carries its subtree's row
    /// offsets in its metadata — a flat integer vector, costing no flatbuffer tables — so planning
    /// reaches the same answer it would have inline, without reading a page.
    #[test]
    fn splits_match_the_inline_layout() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);

            let (inline_segments, inline) = write_chunks(12, 0, &session).await;
            let (paged_segments, paged) = write_chunks(12, 3, &session).await;

            assert_eq!(
                splits_of(&session, paged_segments, &paged),
                splits_of(&session, inline_segments, &inline)
            );
        })
    }

    /// Scan planning is synchronous, so it must reach its answer from the pages' own row counts.
    /// If it ever fetched a page to plan, opening a file would cost every page in it — which is
    /// the cost paging exists to avoid.
    #[test]
    fn planning_does_not_fetch_pages() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let (segments, layout) = write_chunks(12, 3, &session).await;

            let requests = Arc::new(AtomicUsize::new(0));
            let counting = Arc::new(CountingSource {
                inner: segments,
                requests: Arc::clone(&requests),
            });

            let reader = layout
                .new_reader("".into(), counting, &session, &Default::default())
                .unwrap();
            let splits = SplitBy::Layout
                .splits(reader.as_ref(), &(0..12), &[FieldMask::All])
                .unwrap();

            assert_eq!(splits.first(), Some(&0));
            assert_eq!(splits.last(), Some(&12));
            assert_eq!(
                requests.load(Ordering::Relaxed),
                0,
                "planning must not fetch any page segment"
            );
        })
    }

    /// A pruning evaluation that is never awaited must not fetch the page.
    ///
    /// `ZonedReader::pruning_evaluation` builds its data child's pruning future eagerly and only
    /// awaits it when its own zone map did not already prune the range, so a page that issues its
    /// segment request while the future is being constructed is read for ranges that are then
    /// discarded.
    #[test]
    fn pruning_that_is_not_awaited_does_not_fetch_the_page() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let (segments, layout) = write_chunks(12, 3, &session).await;

            let requests = Arc::new(AtomicUsize::new(0));
            let counting = Arc::new(CountingSource {
                inner: segments,
                requests: Arc::clone(&requests),
            });
            let reader = layout
                .new_reader("".into(), counting, &session, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();

            // Built and dropped without ever being polled.
            let _pruning = reader
                .pruning_evaluation(&(0..12), &expr, Mask::new_true(12))
                .unwrap();

            assert_eq!(
                requests.load(Ordering::Relaxed),
                0,
                "a pruning future that was never awaited fetched page segments"
            );
        })
    }

    /// A range that starts and ends inside different pages must still return exactly its rows.
    #[test]
    fn row_range_spanning_pages() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut exec = session.create_execution_ctx();

            // Pages cover 0..3, 3..6, 6..9 and 9..12, so 2..10 clips both end pages.
            let (segments, layout) = write_chunks(12, 3, &session).await;

            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(&(2..10), &expr, MaskFuture::new_true(8))
                .unwrap()
                .await
                .unwrap();

            let expected: PrimitiveArray = (2i32..10).collect();
            assert_arrays_eq!(result, expected.into_array(), &mut exec);
        })
    }

    /// Twelve chunks need thirteen tables inline. Grouped into pages of three, no single
    /// flatbuffer in the file needs more than five — which is the whole point: the verifier's
    /// table limit applies per flatbuffer, and paging bounds every one of them.
    #[test]
    fn paging_keeps_every_flatbuffer_under_a_table_budget() {
        const BUDGET: usize = 8;

        block_on(|handle| async {
            let session = new_session().with_handle(handle);

            let (_, inline) = write_chunks(12, 0, &session).await;
            let inline = to_flatbuffer(&inline);
            assert!(
                verifies_within(&inline, 4 * BUDGET),
                "the inline layout must be a valid flatbuffer given enough tables, \
                 otherwise the budget below proves nothing"
            );
            assert!(
                !verifies_within(&inline, BUDGET),
                "twelve inline chunks should exceed a {BUDGET}-table budget"
            );

            let (segments, paged) = write_chunks(12, 3, &session).await;
            assert!(
                verifies_within(&to_flatbuffer(&paged), BUDGET),
                "a root of four pages should fit a {BUDGET}-table budget"
            );

            // Each page is a flatbuffer root in its own right, verified independently.
            assert_eq!(paged.nchildren(), 4);
            for idx in 0..paged.nchildren() {
                let page = paged.slot(idx).unwrap().unwrap();
                let page = page.as_::<Paged>();
                let bytes = (segments.as_ref() as &dyn SegmentSource)
                    .request(page.segment_id())
                    .await
                    .unwrap()
                    .to_host_sync();
                assert!(
                    verifies_within(bytes.as_ref(), BUDGET),
                    "page {idx} should fit a {BUDGET}-table budget"
                );
            }
        })
    }
}
