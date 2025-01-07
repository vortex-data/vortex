mod scan;

use std::fmt::Debug;
use std::sync::Arc;

pub use scan::*;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::ArrayData;
use vortex_dtype::{DType, FieldPath};
use vortex_error::VortexResult;

use crate::operations::{Operation, Poll};
use crate::segments::SegmentReader;
use crate::{LayoutData, RowMask};

pub type ScanOp = Box<dyn Operation<Output = ArrayData>>;
pub type StatsOp = Box<dyn Operation<Output = Vec<StatsSet>>>;

/// A [`LayoutScan`] provides an encapsulation of an invocation of a scan operation.
pub trait LayoutScan {
    /// Returns the [`LayoutData`] that this scan is operating on.
    fn layout(&self) -> &LayoutData;

    /// The result [`DType`] of the scan after any projections have been applied.
    fn dtype(&self) -> &DType;

    /// Return a [`Scanner`] for the given row mask.
    ///
    /// Note that since a [`Scanner`] returns a single ArrayData, the caller is responsible for
    /// ensuring the working set and result of the scan fit into memory. The [`LayoutData`] can
    /// be asked for "splits" if the caller needs a hint for how to partition the scan.
    fn create_scanner(self: Arc<Self>, mask: RowMask) -> VortexResult<ScanOp>;

    /// Returns a read statistics operation for each requested field path.
    fn create_stats(&self, _field_mask: &[FieldPath], _stats: &[Stat]) -> VortexResult<StatsOp> {
        todo!()
    }
}

pub trait LayoutScanExt: LayoutScan {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutScan>
    where
        Self: Sized + 'static,
    {
        Arc::new(self) as _
    }
}

impl<L: LayoutScan> LayoutScanExt for L {}

/// A scanner with an [`ArrayData`] that is always returned.
#[derive(Debug)]
pub struct ResolvedScanner(pub ArrayData);

impl Operation for ResolvedScanner {
    type Output = ArrayData;

    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        Ok(Poll::Some(self.0.clone()))
    }
}
