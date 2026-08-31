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
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::BoundExpression;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::SharedVortexResult;
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
use crate::layouts::indexed::IndexedLayout;
use crate::layouts::indexed::index::IndexExactness;
use crate::layouts::indexed::index::IndexQueryPlan;
use crate::layouts::indexed::index::RowLocator;
use crate::scan::scan_builder::ScanBuilder;
use crate::segments::SegmentSource;

/// One probe result, shared by every split that needs it.
type SharedProbe = Shared<BoxFuture<'static, SharedVortexResult<Arc<RowLocator>>>>;

/// A reader for the [`crate::layouts::indexed::Indexed`] layout.
///
/// Probes happen once per expression per file: the shared future is cached, and each split slices
/// its own row range out of the resulting locator rather than re-probing.
pub struct IndexedReader {
    layout: IndexedLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    session: VortexSession,
    /// Cached probes keyed by expression. `None` means no index claimed the expression, so the
    /// lookup is not retried.
    probes: DashMap<BoundExpression, Option<CachedProbe>>,
}

#[derive(Clone)]
struct CachedProbe {
    exactness: IndexExactness,
    locator: SharedProbe,
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

        Ok(Self {
            layout,
            name,
            lazy_children,
            session,
            probes: DashMap::default(),
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// Find the first index kind with a claim on `expr` and start (or reuse) its probe.
    ///
    /// One probe per expression per file: every split slices its own row range out of the shared
    /// locator rather than re-probing. The vacant-entry insert holds the shard lock across
    /// planning so two splits racing on the same expression cannot both issue the probe's IO;
    /// planning only touches the child readers, never this map, so it cannot re-enter.
    fn probe(&self, expr: &BoundExpression) -> VortexResult<Option<CachedProbe>> {
        if let Some(cached) = self.probes.get(expr) {
            return Ok(cached.value().clone());
        }

        match self.probes.entry(expr.clone()) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let probe = self.plan_probe(expr)?;
                entry.insert(probe.clone());
                Ok(probe)
            }
        }
    }

    fn plan_probe(&self, expr: &BoundExpression) -> VortexResult<Option<CachedProbe>> {
        for (idx, spec) in self.layout.indexes().iter().enumerate() {
            // Unregistered kinds are inert: their child is never read.
            let Some(vtable) = spec.vtable() else {
                trace!(index = %spec.id(), "index kind not registered, skipping");
                continue;
            };

            let Some(plan) = vtable.plan(expr, self.layout.dtype(), spec.options())? else {
                continue;
            };

            trace!(index = %spec.id(), %expr, filter = %plan.filter, "index claimed expression");

            let index_reader = Arc::clone(self.lazy_children.get(idx + 1)?);
            let exactness = plan.exactness;
            let locator = probe_index(
                index_reader,
                plan,
                self.layout.row_count(),
                self.session.clone(),
            )?;

            return Ok(Some(CachedProbe { exactness, locator }));
        }

        Ok(None)
    }
}

/// Run a plan's filter as a real scan over the index child, then fold the surviving posting rows
/// into a locator.
///
/// Going through [`ScanBuilder`] rather than calling the reader's evaluations directly is what
/// makes the probe cheap. The scan loop splits the index child at its natural chunk boundaries and
/// prunes each split before projecting it, so the sorted key column's zone map narrows the probe to
/// the few chunks that can hold the query's keys and no other posting bytes are ever fetched.
/// Evaluating the whole index child in one call instead would decode every posting list in it.
fn probe_index(
    index_reader: LayoutReaderRef,
    plan: IndexQueryPlan,
    data_row_count: u64,
    session: VortexSession,
) -> VortexResult<SharedProbe> {
    // The index child's dtype is only known once its layout child is materialized, so the plan's
    // filter is bound here rather than by the index kind that produced it.
    let bound_filter = plan.filter.bind(index_reader.dtype())?;
    let postings = ScanBuilder::new(session.clone(), index_reader)
        .with_filter(bound_filter)
        .into_array_stream()?;

    let resolve = Arc::clone(&plan.resolve);
    Ok(async move {
        // Only the rows matching the plan's key predicate survive, one per query term, so
        // collecting them into a single array is cheap regardless of index size.
        let postings: ArrayRef = postings.read_all().await?;
        let mut ctx = session.create_execution_ctx();
        let locator = resolve.resolve(&postings, data_row_count, &mut ctx)?;
        Ok(Arc::new(locator))
    }
    .map_err(Arc::new)
    .boxed()
    .shared())
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

        let Some(probe) = self.probe(expr)? else {
            return Ok(data_eval);
        };

        let row_range = row_range.clone();
        let name = Arc::clone(&self.name);
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let locator = probe.locator.await?;
            let mut result = mask.bitand(&locator.mask_for(&row_range)?);

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
        // An exact index answers the conjunct outright, so the data child is never decoded for it.
        // A superset index can only prune, and the data child re-checks the real predicate. Either
        // way this reuses the cached probe, so a superset conjunct costs no extra IO here.
        if let Some(probe) = self
            .probe(expr)?
            .filter(|probe| probe.exactness == IndexExactness::Exact)
        {
            let row_range = row_range.clone();
            let len = mask.len();
            return Ok(MaskFuture::new(len, async move {
                let locator = probe.locator.await?;
                let index_mask = locator.mask_for(&row_range)?;
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
