mod scan;

use std::fmt::Debug;
use std::sync::Arc;

pub use scan::*;
use vortex_array::stats::{ArrayStatistics, Stat, StatsSet};
use vortex_array::ArrayData;
use vortex_dtype::{DType, FieldPath};
use vortex_error::VortexResult;

use crate::operations::scan::ScanOp;
use crate::operations::stats::StatsOp;
use crate::operations::{Operation, Operator, Poll};
use crate::segments::SegmentReader;
use crate::{LayoutData, RowMask};

/// A [`LayoutScan`] provides an encapsulation of an invocation of a scan operation.
pub trait LayoutScan: 'static + Send + Sync + Debug {
    /// Returns the [`LayoutData`] that this scan is operating on.
    fn layout(&self) -> &LayoutData;

    /// The result [`DType`] of the scan after any projections have been applied.
    fn dtype(&self) -> &DType;

    /// Return a [`Scanner`] for the given row mask.
    ///
    /// Note that since a [`Scanner`] returns a single ArrayData, the caller is responsible for
    /// ensuring the working set and result of the scan fit into memory. The [`LayoutData`] can
    /// be asked for "splits" if the caller needs a hint for how to partition the scan.
    fn create_scanner(self: Arc<Self>, mask: RowMask) -> VortexResult<Box<dyn Operation<ScanOp>>>;

    /// Returns a [`StatsSet`] for each requested field path.
    fn field_stats(
        &self,
        _field_mask: &[FieldPath],
        _stats: &[Stat],
    ) -> VortexResult<Box<dyn Operation<StatsOp>>> {
        todo!()
    }
}

pub trait LayoutScanExt: LayoutScan {
    /// Box the layout scan.
    fn into_arc(self) -> Arc<dyn LayoutScan>
    where
        Self: Sized,
    {
        Arc::new(self) as _
    }
}

impl<L: LayoutScan> LayoutScanExt for L {}

/// A scanner with an [`ArrayData`] that is always returned.
#[derive(Debug)]
pub struct ResolvedScanner(pub ArrayData);

impl Operation<ScanOp> for ResolvedScanner {
    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<ArrayData>> {
        Ok(Poll::Some(self.0.clone()))
    }
}
