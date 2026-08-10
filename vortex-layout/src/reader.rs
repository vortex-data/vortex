// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
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
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::children::LayoutChildren;
use crate::segments::SegmentSource;

/// Shared handle to a stateful layout reader.
pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// A row range used when registering natural scan splits.
///
/// Row range is relative to the reader that receives it. Offset is the offset
/// that the local row range needs to be shifted by to get the global row range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitRange {
    row_offset: u64,
    row_range: Range<u64>,
}

impl SplitRange {
    /// Constructs a split range, returning an error if the local row range is invalid.
    pub fn try_new(row_offset: u64, row_range: Range<u64>) -> VortexResult<Self> {
        if row_range.start > row_range.end {
            vortex_bail!("Invalid split range {:?}", row_range);
        }

        Ok(Self {
            row_offset,
            row_range,
        })
    }

    /// Constructs a split range for the root layout.
    pub fn root(row_range: Range<u64>) -> VortexResult<Self> {
        Self::try_new(0, row_range)
    }

    /// The root-layout row offset of this reader's local row zero.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }

    /// The local row range within this reader.
    pub fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    /// The length of the local row range.
    pub fn len(&self) -> u64 {
        self.row_range.end - self.row_range.start
    }

    /// Returns `true` if the local row range is empty.
    pub fn is_empty(&self) -> bool {
        self.row_range.is_empty()
    }

    /// Returns the equivalent row range in the root layout's coordinate space.
    pub fn root_row_range(&self) -> Range<u64> {
        self.row_offset + self.row_range.start..self.row_offset + self.row_range.end
    }

    /// Returns an error if the local row range is outside the given row count.
    pub fn check_bounds(&self, row_count: u64) -> VortexResult<()> {
        if self.row_range.end > row_count {
            vortex_bail!(
                "Split range {:?} is out of bounds for row count {}",
                self.row_range,
                row_count
            );
        }

        Ok(())
    }
}

/// A collection of root-coordinate row split points.
///
/// Boundaries arrive as non-descending runs, one per layout subtree walked; a descending push
/// starts a new run. A run that exactly repeats the previous surviving run — common when sibling
/// columns share chunk boundaries — is dropped as it completes, and the final sort is skipped
/// when only a single run survives (the boundaries are then already ascending).
pub struct RowSplits {
    splits: Vec<u64>,
    /// Start index of the run currently being appended.
    run_start: usize,
    /// Start index of the surviving run immediately before `run_start`.
    prev_run_start: usize,
}

impl RowSplits {
    /// Add a row boundary to the split set.
    pub fn push(&mut self, row: u64) {
        if let Some(&last) = self.splits.last()
            && row < last
        {
            self.close_run();
        }
        self.splits.push(row);
    }

    /// Close the current run: drop it if it exactly repeats the previous surviving run,
    /// otherwise keep it and start a new run.
    fn close_run(&mut self) {
        if !self.drop_repeated_run() {
            self.prev_run_start = self.run_start;
            self.run_start = self.splits.len();
        }
    }

    /// Drop the current run if it exactly repeats the run before it. Repeated runs contribute
    /// nothing to the sorted-deduped boundary set.
    fn drop_repeated_run(&mut self) -> bool {
        if self.run_start > 0
            && self.splits[self.prev_run_start..self.run_start] == self.splits[self.run_start..]
        {
            self.splits.truncate(self.run_start);
            return true;
        }
        false
    }

    /// Extend with a batch of non-descending row boundaries.
    ///
    /// Only the first element is checked for a descent against the current run, so the caller
    /// must ensure the batch itself is non-descending.
    pub fn extend_ascending(&mut self, rows: impl IntoIterator<Item = u64>) {
        let mut rows = rows.into_iter();
        let Some(first) = rows.next() else {
            return;
        };
        self.push(first);
        self.splits.extend(rows);
        debug_assert!(
            self.splits[self.run_start..].is_sorted(),
            "extend_ascending batch must be non-descending"
        );
    }

    /// Reserve space for additional row boundaries.
    pub fn reserve(&mut self, additional: usize) {
        self.splits.reserve(additional);
    }

    /// Create a new RowSplits with preallocated "capacity"
    pub(crate) fn new_capacity(capacity: usize) -> Self {
        Self {
            splits: Vec::with_capacity(capacity),
            run_start: 0,
            prev_run_start: 0,
        }
    }

    pub(crate) fn into_sorted_deduped(mut self) -> Vec<u64> {
        let final_run_dropped = self.drop_repeated_run();
        // Surviving runs always have a descent between them, so the boundaries are ascending
        // iff a single run survived: no run before the final one (`prev_run_start == 0`) and
        // the final one either is the first (`run_start == 0`) or was dropped.
        let sorted = self.prev_run_start == 0 && (self.run_start == 0 || final_run_dropped);
        if !sorted {
            self.splits.sort_unstable();
        }
        self.splits.dedup();
        self.splits.shrink_to_fit();
        self.splits
    }
}

/// Stateful reader for a [`crate::Layout`].
///
/// A reader owns or references any state needed to evaluate many scan operations over the same
/// layout, such as child readers, decoded metadata, or segment caches. Scan planning calls
/// [`register_splits`](Self::register_splits); execution calls pruning, filter, and projection
/// evaluation for each selected row range.
pub trait LayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns this reader as [`Any`] for downcasting by specialized wrappers.
    fn as_any(&self) -> &dyn Any;

    /// Returns the un-projected dtype of the layout reader.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the layout.
    fn row_count(&self) -> u64;

    /// Register natural split boundaries for this reader.
    ///
    /// `field_mask` contains the projected and filtered field paths needed by the scan.
    /// Implementations should add root-coordinate split boundaries to `splits`, constrained to
    /// `split_range`.
    // TODO(ngates): this is a temporary API until we make layout readers stream based.
    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()>;

    /// Returns a mask where all false values are proven to be false in the given expression.
    ///
    /// The returned mask **does not** need to have been intersected with the input mask.
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
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
        expr: &BoundExpression,
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
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture>;
}

/// Future resolving to a projected Vortex array.
pub type ArrayFuture = BoxFuture<'static, VortexResult<ArrayRef>>;

/// Helpers for futures that resolve to arrays.
pub trait ArrayFutureExt {
    /// Apply a row mask to the resolved array.
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

/// Per-child metadata for [`LazyReaderChildren`].
enum ChildMeta {
    /// Every child shares one dtype and one debug name (e.g. chunked layouts), avoiding a
    /// clone per child at construction.
    Uniform { dtype: DType, name: Arc<str> },
    /// Distinct dtype and name per child.
    PerChild {
        dtypes: Vec<DType>,
        names: Vec<Arc<str>>,
    },
}

/// Lazily constructs and caches child readers while preserving reader context.
pub struct LazyReaderChildren {
    children: Arc<dyn LayoutChildren>,
    meta: ChildMeta,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    ctx: LayoutReaderContext,
    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceCell<LayoutReaderRef>>,
}

impl LazyReaderChildren {
    /// Create a lazy child-reader cache.
    ///
    /// `dtypes` and `names` must be aligned with the child indices exposed by `children`.
    pub fn new(
        children: Arc<dyn LayoutChildren>,
        dtypes: Vec<DType>,
        names: Vec<Arc<str>>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        Self::with_meta(
            children,
            ChildMeta::PerChild { dtypes, names },
            segment_source,
            session,
            ctx,
        )
    }

    /// Create a lazy child-reader cache where every child shares `dtype` and `name`.
    pub fn new_uniform(
        children: Arc<dyn LayoutChildren>,
        dtype: DType,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        Self::with_meta(
            children,
            ChildMeta::Uniform { dtype, name },
            segment_source,
            session,
            ctx,
        )
    }

    fn with_meta(
        children: Arc<dyn LayoutChildren>,
        meta: ChildMeta,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            meta,
            segment_source,
            session,
            ctx,
            cache,
        }
    }

    /// Return the child reader at `idx`, constructing it on first access.
    pub fn get(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }

        self.cache[idx].get_or_try_init(|| {
            let (dtype, name) = match &self.meta {
                ChildMeta::Uniform { dtype, name } => (dtype, name),
                ChildMeta::PerChild { dtypes, names } => (&dtypes[idx], &names[idx]),
            };
            let child = self.children.child(idx, dtype)?;
            child.new_reader(
                Arc::clone(name),
                Arc::clone(&self.segment_source),
                &self.session,
                &self.ctx,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::RowSplits;

    /// The result must always equal the plain sort+dedup of every pushed value, regardless of
    /// how pushes group into runs or which runs get dropped early.
    #[rstest]
    // Identical runs collapse (aligned columns).
    #[case(vec![vec![0, 5, 10], vec![0, 5, 10], vec![0, 5, 10]])]
    // Misaligned runs merge through the sort fallback.
    #[case(vec![vec![0, 5, 10], vec![0, 3, 10]])]
    // A run that is a strict prefix of its predecessor is not dropped.
    #[case(vec![vec![0, 5, 10], vec![0, 5]])]
    // A run extending its predecessor survives.
    #[case(vec![vec![10, 20, 30], vec![10, 20, 30, 40]])]
    // Repeated runs after a distinct run still collapse.
    #[case(vec![vec![0, 5], vec![0, 3, 5], vec![0, 3, 5]])]
    // Identical runs re-diverging.
    #[case(vec![vec![0, 5], vec![0, 5], vec![0, 3]])]
    // Single-element descending runs.
    #[case(vec![vec![5], vec![3], vec![1]])]
    // Adjacent duplicates within a run.
    #[case(vec![vec![0, 5, 5, 10]])]
    // Single run stays untouched.
    #[case(vec![vec![0, 5, 10]])]
    // No pushes at all.
    #[case(vec![])]
    fn into_sorted_deduped_matches_model(#[case] runs: Vec<Vec<u64>>) {
        let mut splits = RowSplits::new_capacity(16);
        let mut model = Vec::new();
        for run in &runs {
            for &row in run {
                splits.push(row);
                model.push(row);
            }
        }
        model.sort_unstable();
        model.dedup();
        assert_eq!(splits.into_sorted_deduped(), model);
    }
}
