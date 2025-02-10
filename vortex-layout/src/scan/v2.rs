use futures::stream;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};

use crate::scan::Scan;

pub trait ScanDriverV2 {}

impl<OldDriver> Scan<OldDriver> {
    /// Perform a scan over our layout.
    ///
    /// The caller (the one who polls the returned ArrayStream) will driver I/O and scheduling
    /// decisions. Note, if the caller is the same async runtime that performs CPU work, then
    /// you should really launch this into a different thread! We can make this ergonomic to do
    /// later.
    pub fn into_stream_v2<D: ScanDriverV2>(self) -> impl ArrayStream + 'static {
        ArrayStreamAdapter::new(self.dtype, stream::empty())
    }
}
