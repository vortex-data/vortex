use vortex_error::VortexResult;

use crate::operations::{Operation, Poll};
use crate::ready;
use crate::segments::SegmentReader;

pub struct CachedOperation<O: Operation> {
    pub(super) op: O,
    pub(super) value: Option<O::Output>,
}

impl<O: Operation> Operation for CachedOperation<O>
where
    O::Output: Clone,
{
    type Output = O::Output;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        if let Some(value) = &self.value {
            return Ok(Poll::Some(value.clone()));
        }

        let value = ready!(self.op.poll(segments));
        self.value.replace(value.clone());
        Ok(Poll::Some(value))
    }
}
