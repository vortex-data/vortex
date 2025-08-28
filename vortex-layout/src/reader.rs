// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::OnceCell;
use vortex_array::stats::Precision;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{vortex_bail, VortexResult};
use vortex_expr::ExprRef;
use vortex_io::runtime::Handle;
use vortex_mask::Mask;

use crate::children::LayoutChildren;
use crate::segments::SegmentSourceRef;

pub type LayoutReaderRef<'rt> = Arc<dyn LayoutReader<'rt> + 'rt>;

/// A [`LayoutReader`] is used to read a [`crate::Layout`] in a way that can cache state across multiple
/// evaluation operations.
pub trait LayoutReader<'rt>: 'rt + Send + Sync {
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
    ) -> VortexResult<Box<dyn PruningEvaluation<'rt>>>;

    /// Performs an exact evaluation of the expression against the layout reader.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation<'rt>>>;

    /// Evaluates the expression against the layout.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation<'rt>>>;
}

/// Returns a mask where all false values are proven to be false in the given expression.
///
/// The returned mask **does not** need to have been intersected with the input mask.
#[async_trait]
pub trait PruningEvaluation<'rt>: 'rt + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

pub struct NoOpPruningEvaluation;

#[async_trait]
impl PruningEvaluation<'_> for NoOpPruningEvaluation {
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
pub trait MaskEvaluation<'rt>: 'rt + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask>;
}

pub struct NoOpMaskEvaluation;

#[async_trait]
impl MaskEvaluation<'_> for NoOpMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        Ok(mask)
    }
}

/// Evaluates an expression against an array, returning an array equal in length to the true count
/// of the input mask.
#[async_trait]
pub trait ArrayEvaluation<'rt>: 'rt + Send + Sync {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef>;
}

pub struct LazyReaderChildren<'rt> {
    children: Arc<dyn LayoutChildren>,
    segment_source: SegmentSourceRef<'rt>,
    handle: Handle<'rt>,

    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceCell<LayoutReaderRef<'rt>>>,
}

impl<'rt> LazyReaderChildren<'rt> {
    pub fn new(
        children: Arc<dyn LayoutChildren>,
        segment_source: SegmentSourceRef<'rt>,
        handle: Handle<'rt>,
    ) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            segment_source,
            cache,
            handle,
        }
    }

    pub fn get(
        &self,
        idx: usize,
        dtype: &DType,
        name: &Arc<str>,
    ) -> VortexResult<&LayoutReaderRef<'rt>> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }

        self.cache[idx].get_or_try_init(|| {
            let child = self.children.child(idx, dtype)?;
            child.new_reader(
                name.clone(),
                self.segment_source.clone(),
                self.handle.clone(),
            )
        })
    }
}
