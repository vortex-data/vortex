use std::ops::Range;
use std::sync::Arc;

use futures_util::stream;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::segments::AsyncSegmentReader;
use vortex_layout::{LayoutData, LayoutReader};
use vortex_scan::Scan;

pub struct VortexFile<R> {
    pub(crate) read: R,
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: Arc<dyn AsyncSegmentReader>,
    // TODO(ngates): not yet used by the file reader
    #[allow(dead_code)]
    pub(crate) splits: Arc<[Range<u64>]>,
}

impl<R> VortexFile<R> {}

/// Async implementation of V ortex File.
impl<R: VortexReadAt> VortexFile<R> {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    /// Returns the DType of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Performs a scan operation over the file.
    pub fn scan(&self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + '_> {
        // Create a shared reader for the scan.
        let reader: Arc<dyn LayoutReader> = self
            .layout
            .reader(self.segments.clone(), self.ctx.clone())?;
        let result_dtype = scan.result_dtype(self.dtype())?;

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let stream = stream::once(async move {
            let row_range = 0..reader.layout().row_count();
            scan.range_scan(row_range)?.evaluate_async(reader).await
        });

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}
