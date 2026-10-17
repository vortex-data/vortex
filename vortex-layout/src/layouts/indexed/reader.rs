//! Read-time probing: answer or prune a conjunct from an index child, else defer to the data
//! child.

use std::any::Any;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::try_join_all;
use itertools::Itertools;
use roaring::RoaringBitmap;
use tracing::trace;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_buffer::BitBufferMut;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::indexed::IndexSpec;
use crate::layouts::indexed::IndexedLayout;
use crate::layouts::indexed::index::IndexExactness;
use crate::layouts::indexed::index::IndexId;
use crate::layouts::indexed::index::IndexVTable;
use crate::layouts::indexed::index::RowLocator;
use crate::scan::split_by::SplitBy;
use crate::segments::SegmentSource;

/// One partition's probe result, shared by every split that needs it.
type SharedProbe = Shared<BoxFuture<'static, SharedVortexResult<Arc<RowLocator>>>>;

/// A reader for the [`crate::layouts::indexed::Indexed`] layout.
///
/// Each index is opened once per reader, which parses its options and caches its decoded chunks,
/// so expressions probing the same chunk share one decode. Probes happen once per expression per
/// index partition: each partition's result is a cached shared future, and every split overlapping
/// that partition slices its own rows out of it rather than re-probing. When more than one spec
/// claims the same expression, the first `Exact` claim covering every partition wins outright;
/// failing that, every claiming spec's locator is kept and intersected at evaluation time.
pub struct IndexedReader {
    layout: IndexedLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    /// The indexes this session can probe, each opened once.
    indexes: Vec<Arc<dyn OpenIndex>>,
    /// Cached claims keyed by expression. `None` means no index claimed the expression, so the
    /// lookup is not retried.
    claims: DashMap<BoundExpression, Option<Claims>>,
}

/// The indexes that claimed one expression.
#[derive(Clone)]
struct Claims {
    /// An exact claim, which answers `filter_evaluation` for any split it fully covers.
    exact: Option<Arc<Claim>>,
    /// Every claim that prunes, intersected by `pruning_evaluation`. An exact claim covering every
    /// partition is the only entry, since nothing else can sharpen it.
    pruning: Vec<Arc<Claim>>,
}

impl IndexedReader {
    pub(crate) fn try_new(
        layout: IndexedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> VortexResult<Self> {
        let mut dtypes = Vec::with_capacity(1 + layout.indexes().len());
        let mut names = Vec::with_capacity(1 + layout.indexes().len());
        dtypes.push(layout.dtype().clone());
        names.push(Arc::clone(&name));
        for spec in layout.indexes().iter() {
            dtypes.push(spec.index_dtype().clone());
            names.push(format!("{}.index:{}", name, spec.id()).into());
        }

        let lazy_children = Arc::new(LazyReaderChildren::new(
            Arc::clone(layout.children()),
            dtypes,
            names,
            segment_source,
            session.clone(),
            ctx,
        ));
        let indexes = open_indexes(&layout, &lazy_children, &session)?;

        Ok(Self {
            layout,
            name,
            lazy_children,
            indexes,
            claims: DashMap::default(),
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// The claims on `expr`, planned once per expression per file.
    ///
    /// The vacant-entry insert holds the shard lock across planning so two splits racing on the
    /// same expression plan it only once; planning only touches the child readers, never this map,
    /// so it cannot re-enter.
    fn claims(&self, expr: &BoundExpression) -> VortexResult<Option<Claims>> {
        if let Some(cached) = self.claims.get(expr) {
            return Ok(cached.value().clone());
        }

        match self.claims.entry(expr.clone()) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let claims = self.plan_claims(expr)?;
                entry.insert(claims.clone());
                Ok(claims)
            }
        }
    }

    fn plan_claims(&self, expr: &BoundExpression) -> VortexResult<Option<Claims>> {
        let mut exact = None;
        let mut pruning = Vec::new();

        for index in &self.indexes {
            let Some(claim) = Arc::clone(index).claim(expr)? else {
                continue;
            };
            let claim = Arc::new(claim);

            if claim.exactness == IndexExactness::Exact {
                if claim.partitions.declined.is_empty() {
                    // Already the best possible answer everywhere: no other spec's claim on this
                    // expression, exact or not, can sharpen it or needs combining with it.
                    return Ok(Some(Claims {
                        exact: Some(Arc::clone(&claim)),
                        pruning: vec![claim],
                    }));
                }
                // Declined partitions leave gaps only other claims can prune, so keep them all.
                exact.get_or_insert_with(|| Arc::clone(&claim));
            }
            pruning.push(claim);
        }

        if pruning.is_empty() {
            return Ok(None);
        }
        Ok(Some(Claims { exact, pruning }))
    }
}

/// Open every index this session can probe, once per reader rather than per expression.
///
/// Unregistered kinds are inert: their child is never read. So is an index whose stored dtype its
/// kind no longer declares, say one written by an incompatible version of it, since its plans
/// would be bound to a schema the child does not have.
fn open_indexes(
    layout: &IndexedLayout,
    children: &Arc<LazyReaderChildren>,
    session: &VortexSession,
) -> VortexResult<Vec<Arc<dyn OpenIndex>>> {
    let mut indexes = Vec::with_capacity(layout.indexes().len());
    for (idx, spec) in layout.indexes().iter().enumerate() {
        let Some(vtable) = spec.vtable() else {
            trace!(index = %spec.id(), "index kind not registered, skipping");
            continue;
        };
        let slot = idx + 1;
        let partitions = Partitions::new(
            spec,
            layout.row_count(),
            layout.children().child_row_count(slot),
        );
        let opened = vtable.open_reader(OpenArgs {
            spec,
            slot,
            dtype: layout.dtype(),
            children: Arc::clone(children),
            partitions: Arc::new(partitions),
            session,
        })?;
        match opened {
            Some(index) => indexes.push(index),
            None => trace!(
                index = %spec.id(),
                stored = %spec.index_dtype(),
                "index dtype does not match what its kind declares, skipping"
            ),
        }
    }
    Ok(indexes)
}

/// What a kind needs to open one of its indexes for reading.
pub(crate) struct OpenArgs<'a> {
    spec: &'a IndexSpec,
    /// The index child's slot in the layout's children.
    slot: usize,
    /// The data child's dtype.
    dtype: &'a DType,
    children: Arc<LazyReaderChildren>,
    partitions: Arc<Partitions>,
    session: &'a VortexSession,
}

/// One index opened for reading, with its options parsed and its decoded chunks cached.
pub(crate) trait OpenIndex: Send + Sync {
    /// This index's claim on `expr`, or `None` if its kind has none.
    fn claim(self: Arc<Self>, expr: &BoundExpression) -> VortexResult<Option<Claim>>;
}

/// Open a spec as a `V`, or `None` if its stored dtype is not what `V` now declares.
pub(crate) fn open_index<V: IndexVTable>(
    vtable: Arc<V>,
    args: OpenArgs<'_>,
) -> VortexResult<Option<Arc<dyn OpenIndex>>> {
    let options = vtable.deserialize_options(args.spec.options())?;
    let index_dtype = args.spec.index_dtype();
    if vtable.index_dtype(args.dtype, &options).as_ref() != Some(index_dtype) {
        return Ok(None);
    }
    Ok(Some(Arc::new(TypedOpenIndex {
        id: args.spec.id(),
        vtable,
        options: Arc::new(options),
        dtype: args.dtype.clone(),
        index_dtype: index_dtype.clone(),
        children: args.children,
        slot: args.slot,
        partitions: args.partitions,
        chunks: DashMap::default(),
        session: args.session.clone(),
    })))
}

/// A decoded chunk of the index child, shared by every probe that reads it.
type SharedChunk<C> = Shared<BoxFuture<'static, SharedVortexResult<Arc<C>>>>;

struct TypedOpenIndex<V: IndexVTable> {
    id: IndexId,
    vtable: Arc<V>,
    options: Arc<V::Options>,
    dtype: DType,
    index_dtype: DType,
    children: Arc<LazyReaderChildren>,
    slot: usize,
    partitions: Arc<Partitions>,
    /// Decoded chunks keyed by their index-child row range, each loaded at most once.
    chunks: DashMap<(u64, u64), SharedChunk<V::Chunk>>,
    session: VortexSession,
}

impl<V: IndexVTable> TypedOpenIndex<V> {
    fn index_reader(&self) -> VortexResult<&LayoutReaderRef> {
        self.children.get(self.slot)
    }

    /// Start (or reuse) loading and decoding the index child's `rows`.
    ///
    /// The future holds only what decoding needs, never `self`, which owns the cache it sits in.
    fn chunk(&self, rows: Range<u64>) -> VortexResult<SharedChunk<V::Chunk>> {
        match self.chunks.entry((rows.start, rows.end)) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let len = usize::try_from(rows.end - rows.start)?;
                let array = self.index_reader()?.projection_evaluation(
                    &rows,
                    &BoundExpression::new_root(self.index_dtype.clone()),
                    MaskFuture::new_true(len),
                )?;
                let vtable = Arc::clone(&self.vtable);
                let options = Arc::clone(&self.options);
                let session = self.session.clone();
                let chunk = async move {
                    let array = array.await?;
                    let mut ctx = session.create_execution_ctx();
                    Ok(Arc::new(vtable.decode(array, &options, &mut ctx)?))
                }
                .map_err(Arc::new)
                .boxed()
                .shared();
                entry.insert(chunk.clone());
                Ok(chunk)
            }
        }
    }
}

impl<V: IndexVTable> OpenIndex for TypedOpenIndex<V> {
    fn claim(self: Arc<Self>, expr: &BoundExpression) -> VortexResult<Option<Claim>> {
        let Some(plan) = self
            .vtable
            .plan(expr, &self.dtype, &self.index_dtype, &self.options)?
        else {
            return Ok(None);
        };
        trace!(index = %self.id, %expr, filter = %plan.filter, "index claimed expression");

        Ok(Some(Claim {
            exactness: plan.exactness,
            partitions: Arc::clone(&self.partitions),
            prober: Arc::new(TypedProber {
                index: self,
                filter: plan.filter,
                query: Arc::new(plan.query),
            }),
            probes: DashMap::default(),
        }))
    }
}

/// Probes one claim's partitions: one expression against one opened index.
trait Prober: Send + Sync {
    fn probe(&self, partition: usize) -> VortexResult<SharedProbe>;
}

struct TypedProber<V: IndexVTable> {
    index: Arc<TypedOpenIndex<V>>,
    filter: BoundExpression,
    query: Arc<V::Query>,
}

impl<V: IndexVTable> Prober for TypedProber<V> {
    /// Prune the partition's index chunks with the plan's filter, so zone maps on the index child
    /// still skip chunks that cannot hold the answer, then resolve the query from the decoded
    /// survivors, decoding each at most once per reader.
    fn probe(&self, partition: usize) -> VortexResult<SharedProbe> {
        let index_rows = self.index.partitions.index_rows(partition);
        let data_rows = self.index.partitions.data_rows(partition);

        let mut chunks = Vec::new();
        if !index_rows.is_empty() {
            let reader = self.index.index_reader()?;
            let bounds = SplitBy::Layout.splits(reader.as_ref(), &index_rows, &[FieldMask::All])?;
            for (start, end) in bounds.into_iter().tuple_windows() {
                let len = usize::try_from(end - start)?;
                let pruned =
                    reader.pruning_evaluation(&(start..end), &self.filter, Mask::new_true(len))?;
                chunks.push((start..end, pruned));
            }
        }

        let index = Arc::clone(&self.index);
        let query = Arc::clone(&self.query);
        Ok(async move {
            let mut surviving = Vec::with_capacity(chunks.len());
            for (rows, pruned) in chunks {
                if !pruned.await?.all_false() {
                    surviving.push(index.chunk(rows)?);
                }
            }
            let decoded = try_join_all(surviving).await?;
            let locator = index.vtable.resolve(
                &query,
                &decoded,
                data_rows.end - data_rows.start,
                &index.options,
            )?;
            Ok(Arc::new(locator))
        }
        .map_err(|err: VortexError| Arc::new(err))
        .boxed()
        .shared())
    }
}

/// Where each of an index's partitions lives, in the data child and in the index child.
///
/// An unpartitioned index is a single partition spanning both children entirely.
pub(crate) struct Partitions {
    len: u64,
    data_row_count: u64,
    index_ends: Arc<[u64]>,
    declined: RoaringBitmap,
}

impl Partitions {
    fn new(spec: &IndexSpec, data_row_count: u64, index_row_count: u64) -> Self {
        match spec.partitioning() {
            Some(partitioning) => Self {
                len: partitioning.partition_len(),
                data_row_count,
                index_ends: partitioning.index_ends().into(),
                declined: partitioning.declined().clone(),
            },
            None => Self {
                // Clamped so an empty data child still divides cleanly.
                len: data_row_count.max(1),
                data_row_count,
                index_ends: Arc::new([index_row_count]),
                declined: RoaringBitmap::new(),
            },
        }
    }

    /// The partitions overlapping `row_range` of the data child.
    fn overlapping(&self, row_range: &Range<u64>) -> Range<usize> {
        let count = self.index_ends.len();
        // Clamped to the partition count, which already fits in memory.
        let clamp = |partition: u64| usize::try_from(partition).map_or(count, |p| p.min(count));
        let end = clamp(row_range.end.div_ceil(self.len));
        clamp(row_range.start / self.len).min(end)..end
    }

    fn data_rows(&self, partition: usize) -> Range<u64> {
        let start = partition as u64 * self.len;
        start..(start + self.len).min(self.data_row_count)
    }

    fn index_rows(&self, partition: usize) -> Range<u64> {
        let start = partition
            .checked_sub(1)
            .map_or(0, |prev| self.index_ends[prev]);
        start..self.index_ends[partition]
    }

    fn is_declined(&self, partition: usize) -> bool {
        u32::try_from(partition).is_ok_and(|partition| self.declined.contains(partition))
    }

    /// Whether every partition overlapping `row_range` was actually built.
    fn covers(&self, row_range: &Range<u64>) -> bool {
        !self
            .overlapping(row_range)
            .any(|partition| self.is_declined(partition))
    }
}

/// One index's claim on one expression, probed a partition at a time as splits need them.
pub(crate) struct Claim {
    exactness: IndexExactness,
    partitions: Arc<Partitions>,
    prober: Arc<dyn Prober>,
    probes: DashMap<usize, SharedProbe>,
}

/// One overlapping partition's contribution to [`Claim::mask`].
enum Part {
    /// Every row in the partition takes this value, without probing.
    Fill(bool),
    Probe(SharedProbe),
    /// Probed only if the selection, once resolved, keeps any of its rows.
    Deferred(usize),
}

impl Claim {
    /// This claim's mask over `row_range` of the data child, stitched from each overlapping
    /// partition's locator. A declined partition proves nothing, so its rows stay set.
    ///
    /// A partition that `selection` rules out entirely is never probed, so its index is never
    /// loaded; its rows come back unset, which callers intersecting with `selection` can't tell
    /// apart from a real answer. Where `selection` has already resolved, as it has for pruning,
    /// probes start here rather than when the future is polled, so a split registers its index IO
    /// as early as its data IO. Otherwise only partitions already probed start now, and the rest
    /// wait on `selection`.
    fn mask(
        self: &Arc<Self>,
        row_range: &Range<u64>,
        selection: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Mask>>> {
        let resolved = selection.clone().now_or_never().transpose()?;

        let mut parts = Vec::new();
        for partition in self.partitions.overlapping(row_range) {
            let rows = self.partitions.data_rows(partition);
            let overlap = row_range.start.max(rows.start)..row_range.end.min(rows.end);
            let local = overlap.start - rows.start..overlap.end - rows.start;
            let selected = usize::try_from(overlap.start - row_range.start)?
                ..usize::try_from(overlap.end - row_range.start)?;

            let part = if self.partitions.is_declined(partition) {
                Part::Fill(true)
            } else if let Some(probe) = self.probes.get(&partition) {
                Part::Probe(probe.clone())
            } else {
                match &resolved {
                    Some(mask) if mask.slice(selected.clone()).all_false() => Part::Fill(false),
                    Some(_) => Part::Probe(self.probe(partition)?),
                    None => Part::Deferred(partition),
                }
            };
            parts.push((local, selected, part));
        }

        let len = usize::try_from(row_range.end - row_range.start)?;
        let claim = Arc::clone(self);
        Ok(async move {
            let mut resolved = resolved;
            let mut bits = BitBufferMut::with_capacity(len);
            for (local, selected, part) in parts {
                let probe = match part {
                    Part::Fill(value) => {
                        bits.append_n(value, usize::try_from(local.end - local.start)?);
                        continue;
                    }
                    Part::Probe(probe) => probe,
                    Part::Deferred(partition) => {
                        let mask = match resolved.take() {
                            Some(mask) => mask,
                            None => selection.clone().await?,
                        };
                        let skip = mask.slice(selected).all_false();
                        resolved = Some(mask);
                        if skip {
                            bits.append_n(false, usize::try_from(local.end - local.start)?);
                            continue;
                        }
                        claim.probe(partition)?
                    }
                };
                probe.await?.append_to(&local, &mut bits)?;
            }
            Ok(Mask::from(bits.freeze()))
        }
        .boxed())
    }

    /// Start (or reuse) the probe of one partition.
    fn probe(&self, partition: usize) -> VortexResult<SharedProbe> {
        match self.probes.entry(partition) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let probe = self.prober.probe(partition)?;
                entry.insert(probe.clone());
                Ok(probe)
            }
        }
    }
}

impl LayoutReader for IndexedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.data_child()?
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        let Some(claims) = self.claims(expr)? else {
            return Ok(data_eval);
        };
        let masks = claims
            .pruning
            .iter()
            .map(|claim| claim.mask(row_range, MaskFuture::ready(mask.clone())))
            .collect::<VortexResult<Vec<_>>>()?;

        let name = Arc::clone(&self.name);
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            // Every claim's mask only narrows what's proven non-matching, so intersect them all;
            // stop as soon as nothing is left alive, rather than awaiting a probe whose answer can
            // no longer change the result.
            let mut result = mask;
            for claim_mask in masks {
                if result.all_false() {
                    break;
                }
                result = result.bitand(&claim_mask.await?);
            }

            // Only bother the data child if the index left anything alive.
            if !result.all_false() {
                result = result.bitand(&data_eval.await?);
            }

            trace!(%name, %expr, density = result.density(), "index pruning evaluation");
            Ok(result)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // Only an exact claim answers the conjunct outright, so the data child is never decoded
        // for it — and only where every overlapping partition was built, since a declined one
        // knows nothing about its rows. A superset claim, however many specs contributed to it,
        // can only prune, so the real predicate always re-checks through the data child. Either
        // way this reuses the cached probes, so a superset conjunct costs no extra IO here.
        if let Some(exact) = self.claims(expr)?.and_then(|claims| claims.exact)
            && exact.partitions.covers(row_range)
        {
            let index_mask = exact.mask(row_range, mask.clone())?;
            let len = mask.len();
            return Ok(MaskFuture::new(len, async move {
                let index_mask = index_mask.await?;
                // Post-condition: the result must be intersected with the input mask.
                Ok(mask.await?.bitand(&index_mask))
            }));
        }

        self.data_child()?.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}
