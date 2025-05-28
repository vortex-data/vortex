use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use vortex_array::stats::Precision;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexError, VortexResult, vortex_bail};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::children::LayoutChildren;
use crate::segments::SegmentSource;

pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// A [`LayoutReader`] is used to read a [`crate::Layout`] in a way that can cache state across multiple
/// evaluation operations.
///
/// It dereferences into the underlying layout being read.
pub trait LayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the un-projected dtype of the layout reader.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the layout reader.
    /// An inexact count may be larger or smaller than the actual row count.
    fn row_count(&self) -> Precision<u64>;

    /// Register the splits of this layout reader.
    // TODO(ngates): this is a temporary API until we make layout readers stream based.
    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;

    /// Performs an approximate evaluation of the expression against the layout reader.
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>>;

    /// Performs an exact evaluation of the expression against the layout reader.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>>;

    /// Evaluates the expression against the layout.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>>;
}

pub type MaskFuture = Shared<BoxFuture<'static, SharedVortexResult<Mask>>>;

/// Create a resolved [`MaskFuture`] from a [`Mask`].
pub fn mask_future_ready(mask: Mask) -> MaskFuture {
    async move { Ok::<_, Arc<VortexError>>(mask) }
        .boxed()
        .shared()
}

#[async_trait]
pub trait PruningEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

pub struct NoOpPruningEvaluation;

#[async_trait]
impl PruningEvaluation for NoOpPruningEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        Ok(mask)
    }
}

/// Refines the given mask, returning a mask equal in length to the input mask.
#[async_trait]
pub trait MaskEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

/// Evaluates an expression against an array, returning an array equal in length to the true count
/// of the input mask.
#[async_trait]
pub trait ArrayEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef>;
}

pub struct LazyReaderChildren {
    children: Arc<dyn LayoutChildren>,
    segment_source: Arc<dyn SegmentSource>,
    ctx: ArrayContext,

    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceLock<LayoutReaderRef>>,
}

impl LazyReaderChildren {
    pub fn new(
        children: Arc<dyn LayoutChildren>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceLock::new()).collect::<Vec<_>>();
        Self {
            children,
            segment_source,
            ctx,
            cache,
        }
    }

    pub fn get(
        &self,
        idx: usize,
        dtype: &DType,
        name: &Arc<str>,
    ) -> VortexResult<&LayoutReaderRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }
        self.cache[idx].get_or_try_init(|| {
            let child = self.children.child(idx, dtype)?;
            child.new_reader(name, &self.segment_source, &self.ctx)
        })
    }
}
