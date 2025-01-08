use vortex_error::{VortexExpect, VortexResult};

use crate::operations::{Operation, Poll};
use crate::ready;
use crate::segments::SegmentReader;

pub struct MapOperation<O, F> {
    pub(super) op: O,
    pub(super) func: Option<F>,
}

impl<R, O, F> Operation for MapOperation<O, F>
where
    O: Operation,
    F: FnOnce(O::Output) -> VortexResult<R>,
{
    type Output = R;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        let v = ready!(self.op.poll(segments));
        let f = self.func.take().vortex_expect("cannot poll Map twice");
        Ok(Poll::Some(f(v)?))
    }
}
