//! The pluggable index-kind contract: what a kind must implement to be built at write time and
//! probed at read time.

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use roaring::RoaringBitmap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

/// Stable registry id of an index kind, e.g. `vortex.idx.reverse_index`.
pub type IndexId = Id;

/// Shared handle to a registered index kind.
pub type IndexVTableRef = Arc<dyn IndexVTable>;

/// A pluggable index kind.
///
/// Mirrors the layout `VTable` machinery: implementations are registered in an
/// [`IndexSession`](crate::layouts::indexed::session::IndexSession) under a stable string id,
/// which is what gets written into the layout metadata. A reader that does not have the kind
/// registered drops the index child and reads the data child directly.
pub trait IndexVTable: 'static + Send + Sync + Debug {
    /// Stable string id, e.g. `vortex.idx.reverse_index`.
    fn id(&self) -> IndexId;

    /// Whether this kind can build an index over values of `dtype`.
    fn supports_dtype(&self, dtype: &DType) -> bool;

    /// Construct a builder for the write path.
    ///
    /// `data_block_len` is the data child's repartition block size when known. Kinds that emit
    /// block-granular locators should default their block length to it so pruned blocks line up
    /// with chunk and segment boundaries.
    fn builder(
        &self,
        dtype: &DType,
        options: &[u8],
        data_block_len: Option<u64>,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn IndexBuilder>>;

    /// Decide whether this index can serve `expr`, a single conjunct scoped to the data child's
    /// dtype.
    ///
    /// `None` means "no claim" and is always safe: the scan falls back to the data child.
    fn plan(
        &self,
        expr: &BoundExpression,
        dtype: &DType,
        options: &[u8],
    ) -> VortexResult<Option<IndexQueryPlan>>;
}

/// Accumulates index content while the data stream is written.
pub trait IndexBuilder: Send {
    /// Chunks arrive in stream order with their absolute row offset within this layout.
    fn push(
        &mut self,
        chunk: &ArrayRef,
        row_offset: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Emit the index content as an array stream, to be written through a child layout strategy.
    ///
    /// Returns the final serialized options alongside it, so builders can record normalization
    /// choices or block sizes discovered during the build.
    ///
    /// `None` declines: nothing worth keeping was built, so no index child and no spec are written,
    /// and the wrapper collapses to the plain data layout if every builder declines. This is the
    /// only point at which size can be judged — a builder is constructed before the first chunk
    /// arrives, so row count and cardinality are not knowable earlier. Declining is always safe: an
    /// absent index reads exactly like an unregistered one.
    fn finish(self: Box<Self>) -> VortexResult<Option<(SendableArrayStream, Vec<u8>)>>;

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

        Ok(Mask::from(bits.freeze()))
    }
}

/// How an index intends to answer one expression.
///
/// The probe runs `filter` as an ordinary scan over the index child — inheriting its zone maps,
/// lazy segment IO and compression — then hands the surviving index rows to `resolve`, which folds
/// them into a locator over the data child's rows.
pub struct IndexQueryPlan {
    /// Whether the resulting mask is exact or a superset.
    pub exactness: IndexExactness,
    /// Predicate over the index child's dtype, selecting the posting rows this query needs.
    ///
    /// Unbound: the index child's dtype is only known once its layout child is materialized, so
    /// the reader binds this against the index child's dtype right before scanning.
    pub filter: Expression,
    /// Folds the selected posting rows into a locator over the data child's row space.
    pub resolve: Arc<dyn IndexResolve>,
}

/// Post-processes probed index rows into a [`RowLocator`].
pub trait IndexResolve: 'static + Send + Sync {
    /// `postings` are the index-child rows that survived [`IndexQueryPlan::filter`], projected in
    /// the index child's own schema.
    fn resolve(
        &self,
        postings: &ArrayRef,
        data_row_count: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RowLocator>;
}
