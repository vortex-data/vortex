use std::sync::Arc;

use futures::Stream;
use vortex_layout::segments::AsyncSegmentReader;

/// A segment source provides a reader for segments, and is then converted into a driver
/// stream that will be polled to make progress.
pub trait SegmentSource {
    fn reader(&self) -> Arc<dyn AsyncSegmentReader>;

    fn into_driver(self) -> impl Stream<Item = ()>;
}
