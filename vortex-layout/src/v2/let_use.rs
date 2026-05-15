// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`LetPlan`] — structural sharing primitive for the layout plan tree.
//!
//! `LetPlan { id, source, body }` registers a [`crate::v2::tee_stream::TeeStream`]
//! over `source` under a stable [`LetId`] in the per-scan
//! [`crate::v2::scan_ctx::ScanCtx`], then delegates execution to `body`.
//! Plans inside `body` (specifically [`UsePlan`]s, built by the CSE
//! pass) subscribe to the tee at execute time and pull chunks at
//! their own pace. The source is polled at most once per scan.
//!
//! Streaming, not broadcast: chunks fan out through the tee as they
//! arrive, sliced per-consumer to that consumer's row range. No
//! whole-array materialisation.
//!
//! See `LAYOUT_PLAN.md` § Tee and CommonSubplanElimination.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scan_ctx::ScanCtxValue;
use crate::v2::tee_stream::TeeStream;

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

/// Per-scan registry of [`LetPlan`] sources, each shared as a
/// [`TeeStream`]. Stored in [`ScanCtx`] under one type slot;
/// lookup is by [`LetId`].
#[derive(Default)]
pub struct LetRegistry {
    streams: HashMap<LetId, Arc<TeeStream>>,
}

impl std::fmt::Debug for LetRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LetRegistry")
            .field("ids", &self.streams.keys().collect::<Vec<_>>())
            .finish()
    }
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
    /// Returns the tee for `id` if one has been registered.
    pub fn get_stream(&self, id: LetId) -> Option<Arc<TeeStream>> {
        self.streams.get(&id).cloned()
    }

    /// Get-or-init: if `id` has not yet been registered, runs `init`
    /// to construct the [`TeeStream`] and stores it. Returns a clone
    /// of the (shared) Arc so multiple consumers can `subscribe`.
    pub fn get_or_init_stream(
        &mut self,
        id: LetId,
        init: impl FnOnce() -> TeeStream,
    ) -> Arc<TeeStream> {
        let arc = self.streams.entry(id).or_insert_with(|| Arc::new(init()));
        Arc::clone(arc)
    }
}

/// Registers a [`TeeStream`] over `source` under [`LetId`], then
/// delegates execution to `body`. Body subtree consumes chunks via
/// [`UsePlan`].
///
/// Pure description — no caches on the plan struct itself; the per-
/// scan tee lives in [`ScanCtx`].
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
    /// pattern is: allocate via [`LetId::fresh`], wire the ID into
    /// the body subtree's lookup site (via [`UsePlan::new`]), then
    /// wrap with `LetPlan::with_id` so the same ID registers the
    /// source.
    pub fn with_id(id: LetId, source: LayoutPlanRef, body: LayoutPlanRef) -> Self {
        Self { id, source, body }
    }

    /// The [`LetId`] under which `source` is registered. Body
    /// subtrees must be constructed with this ID so they can look
    /// up the tee at execute time.
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
        // Idempotent register: first execute installs the tee in
        // the registry; subsequent executes are no-ops. The tee is
        // lazy (doesn't poll source until a UsePlan subscribes).
        drop(self.publish_stream(ctx)?);
        self.body.execute(row_range, ctx)
    }
}

impl LetPlan {
    /// Idempotently register the tee in the scan's [`LetRegistry`].
    /// Returns the shared Arc so callers can subscribe directly if
    /// they want.
    ///
    /// Fast path: read-lock and look up by id. Only when the tee
    /// hasn't been registered do we acquire the write lock and pay
    /// the source-execute cost.
    pub fn publish_stream(&self, ctx: &ScanCtx) -> VortexResult<Arc<TeeStream>> {
        if let Some(registry) = ctx.get_opt::<LetRegistry>()
            && let Some(existing) = registry.get_stream(self.id)
        {
            return Ok(existing);
        }
        // We must call source.execute() before we can hand the
        // resulting stream to TeeStream. Do that under the write
        // lock's get_or_init so two concurrent `publish_stream` calls
        // don't both spin up sources.
        let total_rows = source_total_rows(&self.source);
        let source = Arc::clone(&self.source);
        let ctx_for_init = ctx.clone();
        // `get_or_init_stream` runs `init` at most once per id, so
        // we wrap the fallible source-execute in a `Result` channel
        // via a helper.
        let mut init_err: Option<vortex_error::VortexError> = None;
        let mut registry = ctx.get_mut::<LetRegistry>();
        let arc = registry.get_or_init_stream(self.id, || {
            match source.execute(0..total_rows, &ctx_for_init) {
                Ok(stream) => TeeStream::new(stream),
                Err(e) => {
                    // Stash error for the caller; install an empty
                    // tee so subsequent UsePlans surface clean EOF.
                    init_err = Some(e);
                    TeeStream::new(Box::pin(ArrayStreamAdapter::new(
                        source.schema().clone(),
                        futures::stream::empty(),
                    )))
                }
            }
        });
        if let Some(e) = init_err {
            return Err(e);
        }
        Ok(arc)
    }
}

/// Sum of `source`'s partition row ranges. Drives the "read the
/// entire source" execute call for the underlying tee.
fn source_total_rows(source: &LayoutPlanRef) -> u64 {
    (0..source.partition_count())
        .filter_map(|i| source.partition_stats(i).ok())
        .map(|s| s.row_count())
        .sum()
}

/// Consumer side of [`LetPlan`] sharing. References a registered
/// tee by [`LetId`] and at execute time subscribes for a fresh
/// cursor, then row-range-slices each chunk.
///
/// Built only by the [`crate::v2::cse`] pass — never by
/// `Layout::plan` directly. The `LetPlan` carrying the matching id
/// must dominate every `UsePlan(id)` in the plan tree, otherwise
/// execute fails with an "unregistered LetId" error.
pub struct UsePlan {
    id: LetId,
    output_dtype: DType,
    /// Total row count of the source. Surfaces as the plan's single
    /// partition's row range.
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

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        if row_range.end > self.row_count {
            vortex_bail!(
                "UsePlan::execute row range {row_range:?} exceeds source row count {}",
                self.row_count
            );
        }
        let registry = ctx.get::<LetRegistry>();
        let tee = registry.get_stream(self.id).ok_or_else(|| {
            vortex_error::vortex_err!(
                "UsePlan: no LetPlan with id {} has been registered (CSE pass bug — \
                 every UsePlan must be dominated by a matching LetPlan in the tree)",
                self.id.raw()
            )
        })?;
        // Drop registry guard before subscribing — `subscribe` locks
        // the tee internally, and the registry guard borrows the
        // ScanCtx slot map.
        drop(registry);

        let mut subscriber = tee.subscribe();
        let dtype = self.output_dtype.clone();
        let target_start = row_range.start;
        let target_end = row_range.end;

        // Per-subscriber row cursor: each chunk we pull from the tee
        // covers some consecutive rows starting at `cursor`. We
        // skip chunks entirely below `target_start`, slice chunks
        // straddling the range boundaries, and stop when we've
        // covered `target_end`.
        let stream = try_stream! {
            let mut cursor: u64 = 0;
            while let Some(chunk_res) = subscriber.next().await {
                let chunk = chunk_res?;
                let chunk_len = chunk.len() as u64;
                let chunk_start = cursor;
                let chunk_end = cursor + chunk_len;
                cursor = chunk_end;

                // Entirely before target — drop and continue.
                if chunk_end <= target_start {
                    continue;
                }
                // Entirely after target — stop pulling.
                if chunk_start >= target_end {
                    break;
                }
                // Overlap. Slice if needed; otherwise pass through.
                let slice_start = target_start.saturating_sub(chunk_start);
                let slice_end = (target_end - chunk_start).min(chunk_len);
                let sliced: ArrayRef = if slice_start == 0 && slice_end == chunk_len {
                    chunk
                } else {
                    let s = usize::try_from(slice_start)?;
                    let e = usize::try_from(slice_end)?;
                    chunk.slice(s..e)?
                };
                yield sliced;

                if chunk_end >= target_end {
                    break;
                }
            }
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

#[cfg(test)]
#[allow(deprecated, reason = "tests use to_primitive() to inspect values")]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::StreamExt;
    use futures::stream;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
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
    use super::UsePlan;
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

    async fn collect_i32(mut s: SendableArrayStream) -> VortexResult<Vec<i32>> {
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            let arr = item?;
            let buf = arr.to_primitive().into_buffer::<i32>();
            out.extend(buf.iter().copied());
        }
        Ok(out)
    }

    #[test]
    fn streaming_let_runs_source_once_for_two_uses() -> VortexResult<()> {
        block_on(|_| async move {
            let ctx = ScanCtx::empty();
            let (source, executes) = CountingPlan::new(vec![vec![1, 2], vec![3, 4]]);

            let id = LetId::fresh();
            let body_left: LayoutPlanRef = Arc::new(UsePlan::new(id, dtype(), 4));
            // Use-only body, but Let needs a body that itself doesn't
            // produce data. For this test, treat body as a single-Use
            // and pull two subscribers directly via publish_stream.
            let plan = LetPlan::with_id(id, Arc::clone(&source) as _, body_left);
            let tee = plan.publish_stream(&ctx)?;
            let s1 = tee.subscribe();
            let s2 = tee.subscribe();

            let (r1, r2) = futures::future::join(collect_i32(s1), collect_i32(s2)).await;
            assert_eq!(r1?, vec![1, 2, 3, 4]);
            assert_eq!(r2?, vec![1, 2, 3, 4]);
            assert_eq!(executes.load(Ordering::SeqCst), 1);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn use_plan_slices_to_its_row_range() -> VortexResult<()> {
        block_on(|_| async move {
            let ctx = ScanCtx::empty();
            let (source, _) = CountingPlan::new(vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]]);
            let id = LetId::fresh();
            // Body uses the let; we'll test by executing UsePlan
            // with a subset row range.
            let body: LayoutPlanRef = Arc::new(UsePlan::new(id, dtype(), 8));
            let plan = LetPlan::with_id(id, Arc::clone(&source) as _, Arc::clone(&body));

            // Trigger publication, then execute Use over rows 2..6
            // (i.e. values 3,4,5,6).
            plan.publish_stream(&ctx)?;
            let stream = body.execute(2..6, &ctx)?;
            let values = collect_i32(stream).await?;
            assert_eq!(values, vec![3, 4, 5, 6]);
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
            // Body executed once. Source `execute` was called when
            // we registered the tee, but its output stream wasn't
            // polled because no UsePlan subscribed.
            assert_eq!(body_execs.load(Ordering::SeqCst), 1);
            assert_eq!(source_execs.load(Ordering::SeqCst), 1);
            Ok::<_, VortexError>(())
        })?;
        Ok(())
    }

    #[test]
    fn registry_returns_none_for_unregistered_id() {
        let ctx = ScanCtx::empty();
        let registry = ctx.get::<LetRegistry>();
        assert!(registry.get_stream(LetId(999)).is_none());
    }
}
