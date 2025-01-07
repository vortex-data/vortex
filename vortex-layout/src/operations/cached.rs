use vortex_error::VortexResult;

use crate::operations::{Operation, Poll};
use crate::ready;
use crate::segments::SegmentReader;

/// A trait for operations that are safe to poll multiple times even once a result has already been
/// returned. If an operation incorrectly implements this trait it may panic.
pub unsafe trait CachedOperation: Operation {}

unsafe impl<O> CachedOperation for Box<O>
where
    O: CachedOperation,
    Box<O>: Operation,
{
}

/// An [`Operation`] that caches the result of another operation to allow itself to be polled
/// multiple times.
pub struct OperationCache<O: Operation> {
    pub(super) op: O,
    pub(super) result: Option<O::Output>,
}

impl<O: Operation> Operation for OperationCache<O>
where
    <O as Operation>::Output: Clone,
{
    type Output = O::Output;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        if let Some(r) = &self.result {
            return Ok(Poll::Some(r.clone()));
        }
        let result = ready!(self.op.poll(segments));
        self.result = Some(result.clone());
        Ok(Poll::Some(result))
    }
}

unsafe impl<O: Operation> CachedOperation for OperationCache<O> where <O as Operation>::Output: Clone
{}
