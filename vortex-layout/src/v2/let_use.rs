// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`LetPlan`] — structural sharing primitive for the layout plan tree.
//!
//! `LetPlan { id, source, body }` publishes `source` under a stable
//! [`LetId`] in the per-scan [`crate::v2::scan_ctx::ScanCtx`], then
//! delegates execution to `body`. Plans inside `body` look up the
//! published value by `LetId` and reuse it across calls.
//!
//! Today only the **broadcast** semantics are implemented: `source`
//! is read once per scan (not per `execute` call) into a single
//! [`crate::layouts::SharedArrayFuture`]. Use cases: dict values,
//! file stats, anything that's whole-array-once.
//!
//! See `LAYOUT_PLAN.md` § Tee and CommonSubplanElimination. The
//! streaming variant — for filter mask fan-out and other intra-execute
//! shared cursors — lands in the PR that consumes it; the
//! [`crate::v2::tee_stream`] primitive is already in place to back it.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::layouts::SharedArrayFuture;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scan_ctx::ScanCtxValue;

/// Identifies a [`LetPlan`] within one scan. IDs are globally unique;
/// collisions across scans are harmless (each scan has its own
/// [`ScanCtx`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LetId(u64);

impl LetId {
    /// Allocate a fresh, globally-unique `LetId`.
    pub fn fresh() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Inner numeric value. Use sparingly — intended for debugging /
    /// EXPLAIN output, not for cross-scan correlation.
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Per-scan registry of broadcasted [`LetPlan`] sources. Stored in
/// [`ScanCtx`] under one type slot; lookup is by [`LetId`].
#[derive(Debug, Default)]
pub struct LetRegistry {
    broadcasts: HashMap<LetId, SharedArrayFuture>,
}

impl ScanCtxValue for LetRegistry {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl LetRegistry {
    /// Returns the broadcast for `id` if one has been published.
    pub fn get_broadcast(&self, id: LetId) -> Option<SharedArrayFuture> {
        self.broadcasts.get(&id).cloned()
    }

    /// Get-or-init: if `id` has not yet been published, runs `init`
    /// to construct the future and stores it. Returns a clone of the
    /// stored (shared) future.
    pub fn get_or_init_broadcast(
        &mut self,
        id: LetId,
        init: impl FnOnce() -> SharedArrayFuture,
    ) -> SharedArrayFuture {
        self.broadcasts.entry(id).or_insert_with(init).clone()
    }
}

/// Publishes `source` under [`LetId`], then delegates execution to
/// `body`. Body subtree consumes the published value via
/// [`LetRegistry::get_broadcast`].
///
/// Pure description — no caches on the plan struct itself; the per-
/// scan cache lives in [`ScanCtx`] (see module docs).
pub struct LetPlan {
    id: LetId,
    source: LayoutPlanRef,
    body: LayoutPlanRef,
}

impl LetPlan {
    /// Construct a Let with a fresh ID and the given source/body.
    /// Use [`Self::with_id`] when the body subtree was built against a
    /// pre-allocated [`LetId`].
    pub fn new(source: LayoutPlanRef, body: LayoutPlanRef) -> Self {
        Self::with_id(LetId::fresh(), source, body)
    }

    /// Construct a Let with a caller-supplied [`LetId`]. The typical
    /// pattern is: allocate via [`LetId::fresh`], wire the ID into the
    /// body subtree's lookup site (e.g. `DictDecodePlan::values_let_id`),
    /// then wrap with `LetPlan::with_id` so the same ID publishes the
    /// source.
    pub fn with_id(id: LetId, source: LayoutPlanRef, body: LayoutPlanRef) -> Self {
        Self { id, source, body }
    }

    /// The [`LetId`] under which `source` is published. Body subtrees
    /// must be constructed with this ID so they can look up the
    /// broadcast at execute time.
    pub fn id(&self) -> LetId {
        self.id
    }
}

impl PartialEq for LetPlan {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && crate::v2::plan::plans_eq(&self.source, &other.source)
            && crate::v2::plan::plans_eq(&self.body, &other.body)
    }
}

impl Eq for LetPlan {}

impl std::hash::Hash for LetPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        crate::v2::plan::hash_plan(&self.source, state);
        crate::v2::plan::hash_plan(&self.body, state);
    }
}

impl LayoutPlan for LetPlan {
    fn schema(&self) -> &DType {
        self.body.schema()
    }

    fn partition_count(&self) -> usize {
        self.body.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        self.body.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.body.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        self.body.required_input_ordered()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.body.maintains_input_order()
    }

    fn children(&self) -> &[LayoutPlanRef] {
        // Source and body aren't contiguous in memory — return body
        // alone for the typical pushdown walk. Source is reachable via
        // `LetPlan::source`. Pushdown rules that need to inspect both
        // can downcast.
        std::slice::from_ref(&self.body)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "LetPlan::with_new_children expected 1 child (body), got {}",
                children.len()
            );
        }
        let body = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_error::vortex_err!("LetPlan with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            id: self.id,
            source: Arc::clone(&self.source),
            body,
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        // Publish (idempotent): if some prior execute on this scan
        // already registered the broadcast, this is a no-op clone.
        drop(self.publish_broadcast(ctx));
        self.body.execute(row_range, ctx)
    }
}

impl LetPlan {
    /// Idempotently register the broadcast in the scan's
    /// [`LetRegistry`]. Returns the (shared) future so callers that
    /// want to consume the value directly can.
    ///
    /// Fast path: take a read lock and look up by id. Only when the
    /// broadcast hasn't been registered do we upgrade to a write lock
    /// and pay the build cost. This matters for cases where many
    /// concurrent `execute` calls all hit the same `Let` (e.g., one
    /// `DictDecodePlan` per chunk under a `ChunkedPlan`).
    pub fn publish_broadcast(&self, ctx: &ScanCtx) -> SharedArrayFuture {
        if let Some(registry) = ctx.get_opt::<LetRegistry>()
            && let Some(existing) = registry.get_broadcast(self.id)
        {
            return existing;
        }
        let mut registry = ctx.get_mut::<LetRegistry>();
        let source = Arc::clone(&self.source);
        let ctx_for_init = ctx.clone();
        let dtype = self.source.schema().clone();
        let total_rows = source_total_rows(&self.source);
        registry.get_or_init_broadcast(self.id, move || {
            build_broadcast_future(source, total_rows, dtype, ctx_for_init)
        })
    }
}

/// Sum of `source`'s partition row ranges. Drives the "read the entire
/// source" execute call inside [`build_broadcast_future`].
fn source_total_rows(source: &LayoutPlanRef) -> u64 {
    (0..source.partition_count())
        .filter_map(|i| source.partition_stats(i).ok())
        .map(|s| s.row_count())
        .sum()
}

/// Build a [`SharedArrayFuture`] that executes `source` over its full
/// row range and folds the resulting chunks into one [`ArrayRef`].
fn build_broadcast_future(
    source: LayoutPlanRef,
    total_rows: u64,
    dtype: DType,
    ctx: ScanCtx,
) -> SharedArrayFuture {
    async move {
        let mut stream = source.execute(0..total_rows, &ctx).map_err(Arc::new)?;
        let mut chunks: Vec<ArrayRef> = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.map_err(Arc::new)?);
        }
        let folded = if chunks.len() == 1 {
            chunks
                .into_iter()
                .next()
                .ok_or_else(|| Arc::new(vortex_error::vortex_err!("len-1 vec was empty")))?
        } else if chunks.is_empty() {
            // Empty source — surface as an empty ChunkedArray of the
            // declared dtype so downstream consumers get a real array
            // back (not a panic).
            ChunkedArray::try_new(Vec::new(), dtype.clone())
                .map_err(Arc::new)?
                .into_array()
        } else {
            ChunkedArray::try_new(chunks, dtype.clone())
                .map_err(Arc::new)?
                .into_array()
        };
        Ok(SharedArray::new(folded).into_array())
    }
    .map_err(|e: Arc<VortexError>| e)
    .boxed()
    .shared()
}

/// Consumer side of [`LetPlan`] / `Use` sharing. References a
/// previously-published broadcast by [`LetId`] and at execute time
/// resolves it from the [`crate::v2::let_use::LetRegistry`] in
/// [`ScanCtx`], slicing to the requested row range.
///
/// Built only by the [`crate::v2::cse`] pass — never by `Layout::plan`
/// directly. The `LetPlan` carrying the matching id must dominate
/// every `UsePlan(id)` in the plan tree, otherwise execute fails
/// with an "unregistered LetId" error.
pub struct UsePlan {
    id: LetId,
    output_dtype: DType,
    /// Row count of the broadcast value. Surfaces as the plan's
    /// single partition's row range.
    row_count: u64,
}

impl UsePlan {
    pub fn new(id: LetId, output_dtype: DType, row_count: u64) -> Self {
        Self {
            id,
            output_dtype,
            row_count,
        }
    }

    pub fn id(&self) -> LetId {
        self.id
    }
}

impl PartialEq for UsePlan {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.output_dtype == other.output_dtype
            && self.row_count == other.row_count
    }
}

impl Eq for UsePlan {}

impl std::hash::Hash for UsePlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.output_dtype.hash(state);
        self.row_count.hash(state);
    }
}

impl LayoutPlan for UsePlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("UsePlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("UsePlan has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if row_range.end > self.row_count {
            vortex_bail!(
                "UsePlan::execute row range {row_range:?} exceeds broadcast row count {}",
                self.row_count
            );
        }
        let registry = ctx.get::<LetRegistry>();
        let fut = registry.get_broadcast(self.id).ok_or_else(|| {
            vortex_error::vortex_err!(
                "UsePlan: no LetPlan with id {} has been registered (CSE pass bug — \
                 every UsePlan must be dominated by a matching LetPlan in the tree)",
                self.id.raw()
            )
        })?;
        // Drop the registry guard before awaiting — its `Ref` borrows
        // the underlying DashMap shard.
        drop(registry);

        let dtype = self.output_dtype.clone();
        let stream = async_stream::try_stream! {
            let array = fut.await.map_err(VortexError::from)?;
            let len_u64 = array.len() as u64;
            let start = usize::try_from(row_range.start)?;
            let end = usize::try_from(row_range.end.min(len_u64))?;
            let sliced = if start == 0 && end == array.len() {
                array
            } else {
                array.slice(start..end)?
            };
            yield sliced;
        };
        Ok(Box::pin(vortex_array::stream::ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::StreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::stream::ArrayStreamAdapter;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::stream::SendableArrayStream;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::LetId;
    use super::LetPlan;
    use super::LetRegistry;
    use crate::v2::plan::LayoutPlan;
    use crate::v2::plan::LayoutPlanRef;
    use crate::v2::plan::PartitionStats;
    use crate::v2::scan_ctx::ScanCtx;

    fn dtype() -> DType {
        DType::Primitive(PType::I32, NonNullable)
    }

    /// Test plan that produces a fixed sequence of arrays, counting
    /// every `execute` call so we can assert sharing behaviour.
    struct CountingPlan {
        chunks: Vec<Vec<i32>>,
        row_count: u64,
        executes: Arc<AtomicUsize>,
        dtype: DType,
    }

    impl CountingPlan {
        fn new(chunks: Vec<Vec<i32>>) -> (Arc<Self>, Arc<AtomicUsize>) {
            let row_count = chunks.iter().map(|c| c.len() as u64).sum();
            let executes = Arc::new(AtomicUsize::new(0));
            let plan = Arc::new(Self {
                chunks,
                row_count,
                executes: Arc::clone(&executes),
                dtype: dtype(),
            });
            (plan, executes)
        }
    }

    impl PartialEq for CountingPlan {
        fn eq(&self, other: &Self) -> bool {
            self.chunks == other.chunks
                && self.row_count == other.row_count
                && self.dtype == other.dtype
            // executes is per-instance state; ignore.
        }
    }

    impl Eq for CountingPlan {}

    impl std::hash::Hash for CountingPlan {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.chunks.hash(state);
            self.row_count.hash(state);
            self.dtype.hash(state);
        }
    }

    impl LayoutPlan for CountingPlan {
        fn schema(&self) -> &DType {
            &self.dtype
        }
        fn partition_count(&self) -> usize {
            1
        }
        fn partition_stats(&self, _partition: usize) -> VortexResult<PartitionStats> {
            Ok(PartitionStats::for_range(0..self.row_count))
        }
        fn output_ordered(&self) -> bool {
            true
        }
        fn required_input_ordered(&self) -> Vec<bool> {
            vec![]
        }
        fn maintains_input_order(&self) -> Vec<bool> {
            vec![]
        }
        fn children(&self) -> &[LayoutPlanRef] {
            &[]
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<LayoutPlanRef>,
        ) -> VortexResult<LayoutPlanRef> {
            Ok(self)
        }
        fn execute(
            &self,
            _row_range: std::ops::Range<u64>,
            _ctx: &ScanCtx,
        ) -> VortexResult<SendableArrayStream> {
            self.executes.fetch_add(1, Ordering::SeqCst);
            let arrays: Vec<_> = self
                .chunks
                .iter()
                .map(|c| Ok(PrimitiveArray::from_iter(c.iter().copied()).into_array()))
                .collect();
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                self.dtype.clone(),
                stream::iter(arrays),
            )))
        }
    }

    #[test]
    fn broadcast_runs_source_once_across_many_publishes() -> VortexResult<()> {
        block_on(|_| async move {
            let ctx = ScanCtx::empty();

            let (source, executes) = CountingPlan::new(vec![vec![1, 2], vec![3, 4]]);
            let (body, _) = CountingPlan::new(vec![vec![10, 20]]);
            let plan = LetPlan::new(source, body);

            // Publish three times — source should only execute once.
            let f1 = plan.publish_broadcast(&ctx);
            let f2 = plan.publish_broadcast(&ctx);
            let f3 = plan.publish_broadcast(&ctx);
            drop(f1.await.map_err(VortexError::from)?);
            drop(f2.await.map_err(VortexError::from)?);
            drop(f3.await.map_err(VortexError::from)?);
            assert_eq!(executes.load(Ordering::SeqCst), 1);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn broadcast_folded_into_single_array() -> VortexResult<()> {
        block_on(|_| async move {
            let ctx = ScanCtx::empty();

            let (source, _) = CountingPlan::new(vec![vec![1, 2, 3], vec![4], vec![5, 6]]);
            let (body, _) = CountingPlan::new(vec![vec![]]);
            let plan = LetPlan::new(source, body);

            let f = plan.publish_broadcast(&ctx);
            let arr = f.await.map_err(VortexError::from)?;
            assert_eq!(arr.len(), 6);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn execute_delegates_to_body() -> VortexResult<()> {
        block_on(|_| async move {
            let ctx = ScanCtx::empty();

            let (source, source_execs) = CountingPlan::new(vec![vec![1, 2]]);
            let (body, body_execs) = CountingPlan::new(vec![vec![10, 20, 30]]);
            let plan = LetPlan::new(Arc::clone(&source) as _, Arc::clone(&body) as _);

            let mut stream = plan.execute(0..3, &ctx)?;
            let mut total = 0;
            while let Some(item) = stream.next().await {
                total += item?.len();
            }
            assert_eq!(total, 3);
            // Body executed once (we asked for one execute), source
            // hasn't been awaited yet — broadcast is lazy until
            // someone awaits the future.
            assert_eq!(body_execs.load(Ordering::SeqCst), 1);
            assert_eq!(source_execs.load(Ordering::SeqCst), 0);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn registry_returns_none_for_unregistered_id() {
        let ctx = ScanCtx::empty();
        let registry = ctx.get::<LetRegistry>();
        assert!(registry.get_broadcast(LetId(999)).is_none());
    }
}
