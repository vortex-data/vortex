use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use tokio::runtime::Handle;
use vortex_array::ArrayData;
use vortex_error::{vortex_err, VortexResult};

use crate::exec::ExecDriver;

/// An [`ExecDriver`] implementation that spawns the futures onto a Tokio runtime.
pub struct TokioDriver(pub Handle);

impl ExecDriver for TokioDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<ArrayData>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>> {
        let handle = self.0.clone();

        // This is how many file splits to make progress on at once. While I/O is resolving for
        // the first, we may as well find out the segments required by the next.
        // TODO(ngates): I picked this number somewhat arbitrarily :)
        let concurrency = 2 * handle.metrics().num_workers();

        stream
            .map(move |future| handle.spawn(future))
            .buffered(concurrency)
            .map(|result| match result {
                Ok(result) => result,
                Err(e) => Err(vortex_err!("Failed to join Tokio result {}", e)),
            })
            .boxed()
    }
}
