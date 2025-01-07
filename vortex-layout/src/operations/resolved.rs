use vortex_error::{VortexExpect, VortexResult};

use crate::operations::{Operation, Poll};
use crate::segments::SegmentReader;

pub struct ResolvedOperation<R> {
    pub(super) result: Option<R>,
}

impl<R> Operation for ResolvedOperation<R> {
    type Output = R;

    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        Ok(Poll::Some(self.result.take().vortex_expect(
            "ResolvedOperation::poll called multiple times",
        )))
    }
}
