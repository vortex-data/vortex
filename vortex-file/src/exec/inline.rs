use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use vortex_error::VortexResult;

use crate::exec::ExecDriver;

/// An [`ExecDriver`] implementation that awaits the futures inline using the caller's runtime.
pub struct InlineDriver {
    concurrency: usize,
}

impl InlineDriver {
    pub fn with_concurrency(concurrency: usize) -> Self {
        Self { concurrency }
    }
}

impl ExecDriver for InlineDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<()>>>,
    ) -> BoxStream<'static, VortexResult<()>> {
        stream.buffered(self.concurrency).boxed()
    }
}
