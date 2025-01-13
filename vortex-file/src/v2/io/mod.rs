pub mod file;

use futures::Stream;
use vortex_error::VortexResult;

use crate::v2::segments::SegmentRequest;

/// An I/O driver for executing segment requests.
///
/// Each request contains a [`vortex_layout::segments::SegmentId`] as well as a one-shot callback
/// channel to post back the result.
pub trait IoDriver: 'static {
    fn drive(
        &self,
        stream: impl Stream<Item = SegmentRequest> + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static;
}
