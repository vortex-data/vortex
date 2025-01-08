use std::sync::Arc;

use vortex_array::stats::Stat;
use vortex_array::ArrayData;
use vortex_dtype::{DType, FieldPath};
use vortex_error::VortexResult;

use crate::operations::{Operation, Poll};
use crate::scanner::{EvalOp, StatsOp};
use crate::segments::SegmentReader;
use crate::{LayoutData, RowMask};

/// A [`LayoutReader`] is an instance of a [`LayoutData`] that can cache state across multiple
/// operations.
pub trait LayoutReader {
    /// Returns the [`LayoutData`] of this reader.
    fn layout(&self) -> &LayoutData;

    /// The result [`DType`] of the scan after any projections have been applied.
    fn dtype(&self) -> &DType;

    /// Return a [`Scanner`] for the given row mask.
    ///
    /// Note that since a [`Scanner`] returns a single ArrayData, the caller is responsible for
    /// ensuring the working set and result of the scan fit into memory. The [`LayoutData`] can
    /// be asked for "splits" if the caller needs a hint for how to partition the scan.
    fn create_eval(self: Arc<Self>, mask: RowMask) -> VortexResult<EvalOp>;

    /// Returns a read statistics operation for each requested field path.
    fn create_stats(&self, _field_mask: &[FieldPath], _stats: &[Stat]) -> VortexResult<StatsOp> {
        todo!()
    }
}

pub trait LayoutScanExt: LayoutReader {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutReader>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }
}

impl<L: LayoutReader> LayoutScanExt for L {}

/// A scanner with an [`ArrayData`] that is always returned.
#[derive(Debug)]
pub struct ResolvedScanner(pub ArrayData);

impl Operation for ResolvedScanner {
    type Output = ArrayData;

    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        Ok(Poll::Some(self.0.clone()))
    }
}
