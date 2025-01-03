mod scan;

pub use scan::*;
use vortex_array::ArrayData;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentReader};
use crate::{LayoutData, RowMask};

/// A [`LayoutScan`] provides an encapsulation of an invocation of a scan operation.
pub trait LayoutScan: Send {
    fn boxed(&self) -> Box<dyn LayoutScan>
    where
        Self: 'static,
    {
        Box::new(self.clone())
    }

    /// Returns the [`LayoutData`] that this scan is operating on.
    fn layout(&self) -> &LayoutData;

    /// The result [`DType`] of the scan.
    fn dtype(&self) -> &DType;

    /// Return a [`Scanner`] for the given row mask.
    ///
    /// Note that since a [`Scanner`] returns a single ArrayData, the caller is responsible for
    /// ensuring the working set and result of the scan fit into memory. The [`LayoutData`] can
    /// be asked for "splits" if the caller needs a hint for how to partition the scan.
    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>>;
}

impl<L: LayoutScan> LayoutScan for Box<L> {
    fn layout(&self) -> &LayoutData {
        self.as_ref().layout()
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        self.as_ref().create_scanner(mask)
    }
}

/// The response to polling a scanner.
pub enum Poll {
    /// The next chunk has been read.
    Some(ArrayData),
    /// The scanner requires additional segments before it can make progress.
    NeedMore(Vec<SegmentId>),
}

/// A trait for scanning a single row range of a layout.
pub trait Scanner: Send {
    /// Attempts to return the [`ArrayData`] result of this ranged scan. If the scanner cannot
    /// make progress, it can return a vec of additional data segments using [`Poll::NeedMore`].
    ///
    /// After the poll function has returned an [`ArrayData`], the result of future calls to
    /// ['poll'] are undefined.
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll>;
}
