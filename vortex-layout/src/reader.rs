// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use once_cell::sync::OnceCell;
use vortex_array::ArrayRef;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{SharedVortexResult, VortexError, VortexResult, vortex_bail};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::children::LayoutChildren;
use crate::masks::BoxMaskIterator;
use crate::row_selection::RowSelectionRef;
use crate::segments::SegmentSource;

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
    /// FIXME(ngates): remove this.
    fn row_count(&self) -> Precision<u64>;

    /// Given a [`SlicedSelection`] which can answer range included queryies, returns an iterator of
    /// [`Mask`]s from the layout reader that cover the full range of rows.
    /// These masks are likely to be partitioned in a way that is reasonable efficient for
    /// partitioning evaluation of the [`LayoutReader`] - but there's no guarantee.
    fn row_masks(&self, selection: &RowSelectionRef, field_mask: &[FieldMask]) -> BoxMaskIterator;

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
///
/// ## Post-conditions
///
/// The returned mask **MUST** have been intersected with the input mask.
#[async_trait]
pub trait MaskEvaluation: 'static + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

pub struct NoOpMaskEvaluation;

#[async_trait]
impl MaskEvaluation for NoOpMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        Ok(mask)
    }
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
