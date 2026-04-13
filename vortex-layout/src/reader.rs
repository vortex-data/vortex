// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::try_join;
use once_cell::sync::OnceCell;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::children::LayoutChildren;
use crate::segments::SegmentSource;

pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// A [`LayoutReader`] is used to read a [`crate::Layout`] in a way that can cache state across multiple
/// evaluation operations.
pub trait LayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the un-projected dtype of the layout reader.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the layout.
    fn row_count(&self) -> u64;

    /// Register the splits of this layout reader.
    // TODO(ngates): this is a temporary API until we make layout readers stream based.
    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;

    /// Returns a mask where all false values are proven to be false in the given expression.
    ///
    /// The returned mask **does not** need to have been intersected with the input mask.
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture>;

    /// Refines the given mask, returning a mask equal in length to the input mask.
    ///
    /// It is recommended to defer awaiting the input mask for as long as possible (ideally, after
    /// all I/O is complete). This allows other conjuncts the opportunity to refine the mask as much
    /// as possible before it is used.
    ///
    /// ## Post-conditions
    ///
    /// The returned mask **MUST** have been intersected with the input mask.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture>;

    /// Evaluates an expression against an array.
    ///
    /// It is recommended to defer awaiting the input mask for as long as possible (ideally, after
    /// all I/O is complete). This allows other conjuncts the opportunity to refine the mask as much
    /// as possible before it is used.
    ///
    /// ## Post-conditions
    ///
    /// The returned array **MUST** have length equal to the true count of the input mask.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture>;
}

pub type ArrayFuture = BoxFuture<'static, VortexResult<ArrayRef>>;

pub trait ArrayFutureExt {
    fn masked(self, mask: MaskFuture) -> Self;
}

impl ArrayFutureExt for ArrayFuture {
    /// Returns a new `ArrayFuture` that masks the output with a mask
    fn masked(self, mask: MaskFuture) -> Self {
        Box::pin(async move {
            let (array, mask) = try_join!(self, mask)?;
            array.mask(mask.into_array())
        })
    }
}

pub struct LazyReaderChildren {
    children: Arc<dyn LayoutChildren>,
    dtypes: Vec<DType>,
    names: Vec<Arc<str>>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceCell<LayoutReaderRef>>,
}

impl LazyReaderChildren {
    pub fn new(
        children: Arc<dyn LayoutChildren>,
        dtypes: Vec<DType>,
        names: Vec<Arc<str>>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            dtypes,
            names,
            segment_source,
            session,
            cache,
        }
    }

    pub fn get(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }

        self.cache[idx].get_or_try_init(|| {
            let dtype = &self.dtypes[idx];
            let child = self.children.child(idx, dtype)?;
            child.new_reader(
                Arc::clone(&self.names[idx]),
                Arc::clone(&self.segment_source),
                &self.session,
            )
        })
    }
}
