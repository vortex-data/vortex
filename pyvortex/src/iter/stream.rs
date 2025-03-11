use futures::StreamExt;
use tokio::runtime::Handle;
use vortex::ArrayRef;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::iter::ArrayIterator;
use vortex::stream::ArrayStream;

use crate::TOKIO_RUNTIME;

pub(crate) trait AsyncRuntime {
    fn block_on<F: Future>(&self, fut: F) -> F::Output;
}

impl AsyncRuntime for Handle {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.block_on(fut)
    }
}

/// Adapter for converting an [`ArrayStream`] into an [`ArrayIterator`].
pub(crate) struct ArrayStreamToIterator<S, AR> {
    stream: S,
    runtime: AR,
}

impl<S: ArrayStream + Unpin + Send> ArrayStreamToIterator<S, Handle> {
    pub(crate) fn new(stream: S) -> Self {
        Self {
            stream,
            runtime: TOKIO_RUNTIME.handle().clone(),
        }
    }
}

impl<S, AR> ArrayIterator for ArrayStreamToIterator<S, AR>
where
    S: ArrayStream + Unpin + Send,
    AR: AsyncRuntime,
{
    fn dtype(&self) -> &DType {
        self.stream.dtype()
    }
}

impl<S, AR> Iterator for ArrayStreamToIterator<S, AR>
where
    S: ArrayStream + Unpin + Send,
    AR: AsyncRuntime,
{
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.runtime.block_on(self.stream.next())
    }
}
