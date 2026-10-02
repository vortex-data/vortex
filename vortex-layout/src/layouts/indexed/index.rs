//! The pluggable index-kind contract: what a kind must implement to be built at write time and
//! probed at read time.

use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

use crate::layouts::indexed::reader::OpenArgs;
use crate::layouts::indexed::reader::OpenIndex;
use crate::layouts::indexed::reader::open_index;
use crate::layouts::indexed::writer::IndexWriter;
use crate::layouts::indexed::writer::TypedIndexWriter;

/// Stable registry id of an index kind, e.g. `vortex.idx.reverse_index`.
pub type IndexId = Id;

/// A pluggable index kind.
///
/// Mirrors the layout `VTable` machinery: implementations are registered in an
/// [`IndexSession`](crate::layouts::indexed::session::IndexSession) under a stable string id,
/// which is what gets written into the layout metadata. A reader that does not have the kind
/// registered drops the index child and reads the data child directly.
///
/// The associated types are what keeps repeated work out of the read path. A reader parses a
/// spec's [`Options`](Self::Options) once, and decodes each chunk of the index child into a
/// [`Chunk`](Self::Chunk) at most once, sharing it between every expression that probes the
/// chunk. Each expression is reduced by [`plan`](Self::plan) to a [`Query`](Self::Query) that
/// [`resolve`](Self::resolve) answers against those decoded chunks.
///
/// # Serialization
///
/// A kind's on-disk format is exactly three things, and nothing else a kind does reaches the file:
///
/// - [`serialize_options`](Self::serialize_options) and
///   [`deserialize_options`](Self::deserialize_options), for the options blob stored in the spec.
///   The blob is the kind's to version.
/// - [`index_dtype`](Self::index_dtype), the schema of the index child.
/// - [`encode`](Self::encode) and [`decode`](Self::decode), between a [`Chunk`](Self::Chunk) and
///   rows of that schema.
///
/// Builders produce chunks and options, never bytes, so these pairs can be tested without
/// writing a layout. Re-encoding a decoded chunk must reproduce its encoding.
pub trait IndexVTable: 'static + Send + Sync + Debug {
    /// The kind's options, parsed from the self-versioned options blob.
    type Options: 'static + Send + Sync;
    /// Builds the index on the write path.
    type Builder: IndexBuilder<Options = Self::Options, Chunk = Self::Chunk> + 'static;
    /// What [`plan`](Self::plan) reduced one expression to, answered by
    /// [`resolve`](Self::resolve).
    type Query: 'static + Send + Sync;
    /// One chunk of the index child, decoded into whatever form the kind answers queries from.
    type Chunk: 'static + Send + Sync;

    /// Stable string id, e.g. `vortex.idx.reverse_index`.
    fn id(&self) -> IndexId;

    /// Serialize options into the blob stored in the spec.
    fn serialize_options(&self, options: &Self::Options) -> Vec<u8>;

    /// Parse an options blob, as configured on the write side or stored in a spec.
    fn deserialize_options(&self, options: &[u8]) -> VortexResult<Self::Options>;

    /// The dtype of the index child this kind builds over values of `dtype`, or `None` if it
    /// cannot index them.
    ///
    /// This is the index's schema: the writer rejects a builder whose output differs from it, and
    /// the reader skips an index whose stored dtype no longer matches it.
    fn index_dtype(&self, dtype: &DType, options: &Self::Options) -> Option<DType>;

    /// Construct a builder for the write path.
    ///
    /// `data_block_len` is the data child's repartition block size when known. Kinds that emit
    /// block-granular locators should default their block length to it so pruned blocks line up
    /// with chunk and segment boundaries.
    ///
    /// A partitioned index calls this once per partition, and each builder sees only its
    /// partition's rows.
    fn builder(
        &self,
        dtype: &DType,
        options: &Self::Options,
        data_block_len: Option<u64>,
        session: &VortexSession,
    ) -> VortexResult<Self::Builder>;

    /// Decide whether this index can serve `expr`, a single conjunct scoped to the data child's
    /// `dtype`.
    ///
    /// `index_dtype` is the index child's dtype, as returned by [`IndexVTable::index_dtype`], and
    /// the scope the plan's filter must be bound to.
    ///
    /// `None` means "no claim" and is always safe: the scan falls back to the data child.
    fn plan(
        &self,
        expr: &BoundExpression,
        dtype: &DType,
        index_dtype: &DType,
        options: &Self::Options,
    ) -> VortexResult<Option<IndexQueryPlan<Self::Query>>>;

    /// Encode a chunk a builder produced as rows of the index child, of the declared
    /// [`index_dtype`](Self::index_dtype).
    fn encode(
        &self,
        chunk: &Self::Chunk,
        options: &Self::Options,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Decode rows of the index child, in its own schema, back into a chunk.
    ///
    /// The index child's layout strategy may chunk it differently than it was encoded, so this
    /// must accept any contiguous run of encoded rows. Called at most once per chunk per reader,
    /// however many expressions probe it.
    fn decode(
        &self,
        chunk: ArrayRef,
        options: &Self::Options,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self::Chunk>;

    /// Answer `query` from one partition's decoded chunks, as a locator local to the partition.
    ///
    /// `chunks` are in index-row order and omit every chunk the plan's
    /// [`filter`](IndexQueryPlan::filter) ruled out, so they hold everything the query can match
    /// in this partition. `data_row_count` is the partition's row count in the data child.
    fn resolve(
        &self,
        query: &Self::Query,
        chunks: &[Arc<Self::Chunk>],
        data_row_count: u64,
        options: &Self::Options,
    ) -> VortexResult<RowLocator>;
}

/// Shared handle to a registered index kind, erasing its [`IndexVTable`] types.
#[derive(Clone)]
pub struct IndexVTableRef(Arc<dyn DynIndexVTable>);

impl IndexVTableRef {
    /// Erase an index kind so it can be registered and configured alongside others.
    pub fn new<V: IndexVTable>(vtable: V) -> Self {
        Self(Arc::new(vtable))
    }

    /// The kind's stable id.
    pub fn id(&self) -> IndexId {
        self.0.id()
    }

    /// The concrete kind, if it is a `V`.
    pub fn as_opt<V: IndexVTable>(&self) -> Option<&V> {
        self.0.as_any().downcast_ref()
    }

    /// Parse `options` once for writing.
    pub(crate) fn open_writer(&self, options: &[u8]) -> VortexResult<Arc<dyn IndexWriter>> {
        Arc::clone(&self.0).open_writer(options)
    }

    /// Parse a spec's options once for reading, or `None` if its stored dtype is not what the
    /// kind now declares.
    pub(crate) fn open_reader(&self, args: OpenArgs<'_>) -> VortexResult<Option<Arc<dyn OpenIndex>>> {
        Arc::clone(&self.0).open_reader(args)
    }
}

impl Debug for IndexVTableRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The type-erased side of [`IndexVTable`], implemented for every kind.
///
/// Kept crate-private so the reader and writer can hand out their own typed state without it
/// becoming public API; kinds only ever implement [`IndexVTable`].
pub(crate) trait DynIndexVTable: 'static + Send + Sync + Debug {
    fn id(&self) -> IndexId;

    fn as_any(&self) -> &dyn Any;

    fn open_writer(self: Arc<Self>, options: &[u8]) -> VortexResult<Arc<dyn IndexWriter>>;

    fn open_reader(self: Arc<Self>, args: OpenArgs<'_>) -> VortexResult<Option<Arc<dyn OpenIndex>>>;
}

impl<V: IndexVTable> DynIndexVTable for V {
    fn id(&self) -> IndexId {
        IndexVTable::id(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn open_writer(self: Arc<Self>, options: &[u8]) -> VortexResult<Arc<dyn IndexWriter>> {
        Ok(Arc::new(TypedIndexWriter::try_new(self, options)?))
    }

    fn open_reader(self: Arc<Self>, args: OpenArgs<'_>) -> VortexResult<Option<Arc<dyn OpenIndex>>> {
        open_index(self, args)
    }
}

/// Accumulates index content while the data stream is written.
pub trait IndexBuilder: Send {
    /// The kind's options, as finally chosen by the build.
    type Options;
    /// The content the kind's [`IndexVTable::encode`] writes out.
    type Chunk;

    /// Chunks arrive in stream order with their row offset within this builder's partition, which
    /// for an unpartitioned index is the whole layout.
    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Emit the index content as chunks, which the writer encodes and writes through a child
    /// layout strategy in order.
    ///
    /// Returns the final options alongside them, so builders can record normalization choices or
    /// block sizes discovered during the build.
    ///
    /// `None` declines: nothing worth keeping was built, so no index child and no spec are written,
    /// and the wrapper collapses to the plain data layout if every builder declines. This is the
    /// only point at which size can be judged — a builder is constructed before the first chunk
    /// arrives, so row count and cardinality are not knowable earlier. Declining is always safe: an
    /// absent index reads exactly like an unregistered one.
    fn finish(self) -> VortexResult<Option<(Vec<Self::Chunk>, Self::Options)>>;

    /// Bytes currently buffered, reported up through the write context's buffered-bytes tracker.
    fn buffered_bytes(&self) -> u64;
}

/// What a probe result means and how precisely it locates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IndexExactness {
    /// True bits are exactly the matching rows, so the probe may serve `filter_evaluation`
    /// directly and skip decoding the data child for that conjunct.
    Exact,
    /// False bits are proven non-matching; true bits are "maybe". Serves `pruning_evaluation`
    /// only, and the real predicate re-checks the survivors.
    Superset,
}

/// Where an index located its matches, in the data child's row space.
///
/// Both variants are roaring bitmaps: postings are naturally set-like, intersect and union
/// cheaply, and expand into a [`Mask`] by walking runs in sorted order.
///
/// Roaring bitmaps are `u32`-keyed, which caps a single layout at `u32::MAX` rows. That is far
/// above any practical Vortex file, and [`RowLocator::mask_for`] errors rather than truncating if
/// it is ever exceeded.
#[derive(Clone, Debug)]
pub enum RowLocator {
    /// Row positions local to the data child.
    Rows(RoaringBitmap),
    /// Ids of fixed `block_len`-row blocks, expanded by broadcasting each block's bit across its
    /// rows — the same shape as the zoned reader's per-zone expansion.
    Blocks { block_len: u64, ids: RoaringBitmap },
}

impl RowLocator {
    /// An empty locator: nothing matches, so everything prunes.
    pub fn empty_rows() -> Self {
        RowLocator::Rows(RoaringBitmap::new())
    }

    /// Expand this locator into a mask covering `row_range` of the data child.
    ///
    /// The returned mask has length `row_range.len()` and is *not* intersected with any input
    /// mask; callers do that.
    pub fn mask_for(&self, row_range: &Range<u64>) -> VortexResult<Mask> {
        let len = usize::try_from(row_range.end - row_range.start)?;
        let mut bits = BitBufferMut::with_capacity(len);
        self.append_to(row_range, &mut bits)?;
        Ok(Mask::from(bits.freeze()))
    }

    /// Append this locator's bits for `row_range` to `bits`, one per row.
    pub fn append_to(&self, row_range: &Range<u64>, bits: &mut BitBufferMut) -> VortexResult<()> {
        match self {
            // Walk the set bits in ascending order, emitting the false run before each one. The
            // bitmap is sorted, so this is a single linear pass with no random access.
            RowLocator::Rows(rows) => {
                let start = u32::try_from(row_range.start)?;
                let end = u32::try_from(row_range.end)?;
                let mut pos = row_range.start;
                for row in rows.range(start..end) {
                    let row = u64::from(row);
                    bits.append_n(false, usize::try_from(row - pos)?);
                    bits.append_n(true, 1);
                    pos = row + 1;
                }
                bits.append_n(false, usize::try_from(row_range.end - pos)?);
            }
            // Broadcast each block's bit across the rows it covers, clipped to `row_range`.
            RowLocator::Blocks { block_len, ids } => {
                let mut row = row_range.start;
                while row < row_range.end {
                    let block = row / block_len;
                    let block_end = ((block + 1) * block_len).min(row_range.end);
                    let hit = ids.contains(u32::try_from(block)?);
                    bits.append_n(hit, usize::try_from(block_end - row)?);
                    row = block_end;
                }
            }
        }
        Ok(())
    }
}

/// How an index intends to answer one expression.
///
/// The probe prunes the index child's chunks with `filter`, inheriting its zone maps, then decodes
/// the surviving chunks (each at most once per reader) and hands them to
/// [`IndexVTable::resolve`] to answer `query`.
pub struct IndexQueryPlan<Q> {
    /// Whether the resulting mask is exact or a superset.
    pub exactness: IndexExactness,
    /// Predicate bound to the index child's dtype. Chunks it proves hold no matching index rows
    /// are neither loaded nor decoded.
    pub filter: BoundExpression,
    /// What [`IndexVTable::resolve`] answers from the decoded chunks.
    pub query: Q,
}
