// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use crate::children::LayoutChildren;
use crate::segments::{SegmentId, SegmentSource, Segments};
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use vortex_array::ArrayRef;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_bail};
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_set::HashSet;

pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// A [`LayoutReader`] is used to read a [`crate::Layout`] in a way that can cache state across multiple
/// evaluation operations.
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

/// Returns a mask where all false values are proven to be false in the given expression.
///
/// The returned mask **does not** need to have been intersected with the input mask.
pub trait PruningEvaluation: 'static + Send + Sync {
    fn invoke(&self, mask: Mask, segments: &dyn Segments) -> VortexResult<Mask>;

    fn required_segments(&self, segments: &mut HashSet<SegmentId>);
}

pub struct NoOpPruningEvaluation;

impl PruningEvaluation for NoOpPruningEvaluation {
    fn invoke(&self, mask: Mask, _segments: &dyn Segments) -> VortexResult<Mask> {
        Ok(mask)
    }

    fn required_segments(&self, _segments: &mut HashSet<SegmentId>) {}
}

/// Refines the given mask, returning a mask equal in length to the input mask.
///
/// ## Post-conditions
///
/// The returned mask **MUST** have been intersected with the input mask.
pub trait MaskEvaluation: 'static + Send + Sync {
    fn invoke(&self, mask: Mask, segments: &dyn Segments) -> VortexResult<Mask>;

    fn required_segments(&self, segments: &mut HashSet<SegmentId>);
}

pub struct NoOpMaskEvaluation;

impl MaskEvaluation for NoOpMaskEvaluation {
    fn invoke(&self, mask: Mask, _segments: &dyn Segments) -> VortexResult<Mask> {
        Ok(mask)
    }

    fn required_segments(&self, _segments: &mut HashSet<SegmentId>) {}
}

/// Evaluates an expression against an array, returning an array equal in length to the true count
/// of the input mask.
pub trait ArrayEvaluation: 'static + Send + Sync {
    fn invoke(&self, mask: Mask, segments: &dyn Segments) -> VortexResult<ArrayRef>;

    fn required_segments(&self, segments: &mut HashSet<SegmentId>);
}

/// Provides semantics equivalent to `LazyLock`, except where segments are bound late.
#[derive(Clone)]
pub struct LazyWithSegments<T>(Arc<RwLock<LazyWithSegmentsInner<T>>>);

struct LazyWithSegmentsInner<T> {
    result: Option<SharedVortexResult<T>>,
    ctor: Option<Box<dyn FnOnce(&dyn Segments) -> VortexResult<T> + Send + Sync>>,
    required_segments: HashSet<SegmentId>,
}

impl<T: Send + Clone> LazyWithSegments<T> {
    pub fn new(
        ctor: impl FnOnce(&dyn Segments) -> VortexResult<T> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(RwLock::new(LazyWithSegmentsInner {
            result: None,
            ctor: Some(Box::new(ctor)),
            required_segments: Default::default(),
        })))
    }

    pub fn with_required_segments(self, segment_ids: impl IntoIterator<Item = SegmentId>) -> Self {
        self.0
            .write()
            .required_segments
            .extend(segment_ids.into_iter());
        self
    }

    pub fn with_lazy_required_segments<R>(self, lazy: &LazyWithSegments<R>) -> Self {
        self.0
            .write()
            .required_segments
            .extend(lazy.0.read().required_segments.iter().cloned());
        self
    }

    pub fn with_array_evaluation_segments(self, eval: &dyn ArrayEvaluation) -> Self {
        eval.required_segments(&mut self.0.write().required_segments);
        self
    }

    pub fn with_mask_evaluation_segments(self, eval: &dyn MaskEvaluation) -> Self {
        eval.required_segments(&mut self.0.write().required_segments);
        self
    }

    pub fn get(&self, segments: &dyn Segments) -> SharedVortexResult<T> {
        {
            let read = self.0.read();
            if let Some(result) = &read.result {
                return result.clone();
            }
        }

        let mut write = self.0.write();
        if let Some(result) = &write.result {
            return result.clone();
        }

        let ctor = write.ctor.take().expect("Constructor already consumed");
        write.result = Some(ctor(segments).map_err(Arc::new));
        write.result.as_ref().cloned().vortex_expect("infallible")
    }

    pub fn required_segments(&self, segments: &mut HashSet<SegmentId>) {
        segments.extend(self.0.read().required_segments.iter().cloned());
    }
}

pub struct LazyReaderChildren {
    children: Arc<dyn LayoutChildren>,
    segment_source: Arc<dyn SegmentSource>,

    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceCell<LayoutReaderRef>>,
}

impl LazyReaderChildren {
    pub fn new(children: Arc<dyn LayoutChildren>, segment_source: Arc<dyn SegmentSource>) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            segment_source,
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
            child.new_reader(name.clone(), self.segment_source.clone())
        })
    }
}
