pub mod file;

use futures::Stream;
use vortex_error::VortexResult;

use crate::segments::SegmentRequest;

/// An I/O driver for executing segment requests.
///
/// Each request contains a [`vortex_layout::segments::SegmentId`] as well as a one-shot callback
/// channel to post back the result.
///
/// I/O drivers are able to coalesce, debounce, or otherwise group the requests, as well as control the concurrency
/// of the I/O operations with [`buffered`](`futures::stream::StreamExt::buffered`).
pub trait IoDriver: 'static {
    // NOTE(ngates): this isn't an async_trait since it doesn't need to be object-safe or boxed.
    fn drive(
        &self,
        stream: impl Stream<Item = SegmentRequest> + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static;
}
