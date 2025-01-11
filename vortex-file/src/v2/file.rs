use std::future::ready;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures_util::{stream, FutureExt, StreamExt, TryFutureExt};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::{ExprEvaluator, LayoutData, LayoutReader};
use vortex_scan::Scan;

use crate::v2::driver::Driver;

pub struct VortexFile {
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) driver: Arc<dyn Driver<ArrayData>>,
    pub(crate) splits: Arc<[Range<u64>]>,
}

/// Async implementation of Vortex File.
impl VortexFile {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    /// Returns the DType of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Performs a scan operation over the file.
    ///
    /// The general architecture of a scan is to spawn a task for each split to be read.
    ///
    /// FIXME(ngates): Ok, so ideally this is the async implementation of scanning. The assumption
    ///  I think is that I/O is driven from the caller by polling the stream, and so we should
    ///  provide the option to configure where and how the CPU tasks are spawned.
    ///
    ///  For DataFusion, we should support the ability to spawn tasks back onto the DataFusion
    ///  runtime, which for some reason it intertwined with I/O.
    ///
    ///  For the sync implementation, I'm guessing we'd want each thread in the pool to perform
    ///  their own synchronous I/O? Otherwise we'd put it all on a single thread and it would
    ///  be slow.
    ///
    ///  So. What do we actually want to be configurable?
    ///    - The number of threads in the row group pool? Or is this behind a spawn abstraction?
    ///    - The number of row groups to concurrently spawn? We could also just provide a stream
    ///      and allow the consumer to buffer it as they see fit.
    ///
    ///  I think there's a mode here which is where we give an AsyncSegmentReader to each task
    ///  and they all put their segment requests onto a single stream. We can call this a
    ///  UnifiedSegmentReader? In this world, all segment requests appear in the same stream
    ///  and can be handled however the consumer sees fit. Out of this comes some stream that we
    ///  can poll indefinitely to drive the coalesced I/O.
    ///
    ///  But what if we don't want coalesced I/O? For whatever reason... we just want to take
    ///  whatever segments are requested and read them immediately. (TODO: we should change
    ///  AsyncSegmentReader to take/return multiple. Maybe a readv kind of thing?). This could
    ///  be useful in a thread-per-core model because it could wrap I/O-uring on the same thread.
    ///
    ///  The API must therefore be to have a `Driver` trait that takes and returns a stream and it
    ///  can process it as it sees fit.
    pub fn scan(self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + 'static> {
        // Create a shared reader for the scan.
        let reader: Arc<dyn LayoutReader> =
            self.layout.reader(self.driver.reader(), self.ctx.clone())?;

        // Iterate each split, and evaluate its range scan.
        let stream = stream::iter(ArcIter::new(self.splits.clone())).map(move |row_range| {
            let reader = reader.clone();
            ready(scan.clone().range_scan(row_range))
                .and_then(|range_scan| {
                    range_scan.evaluate_async(|row_mask, expr| reader.evaluate_expr(row_mask, expr))
                })
                .boxed()
                .unwrap_or_else(|_cancelled| Err(vortex_err!("recv failed, send dropped")))
        });

        // Wrap up the stream with the driver
        let stream = self.driver.drive(stream.boxed());

        let result_dtype = scan.result_dtype(self.dtype())?;

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}

/// There is no `IntoIterator` for `Arc<[T]>` so to avoid copying into a Vec<T>, we define our own.
/// See <https://users.rust-lang.org/t/arc-to-owning-iterator/115190/11>.
struct ArcIter<T> {
    inner: Arc<[T]>,
    pos: usize,
}

impl<T> ArcIter<T> {
    fn new(inner: Arc<[T]>) -> Self {
        Self { inner, pos: 0 }
    }
}

impl<T: Clone> Iterator for ArcIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.pos < self.inner.len()).then(|| {
            let item = self.inner[self.pos].clone();
            self.pos += 1;
            item
        })
    }
}
