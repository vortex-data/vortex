use std::future::ready;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use tokio::runtime::Handle;
use vortex_array::ArrayData;
use vortex_error::{vortex_err, VortexResult};

use crate::exec::ExecDriver;

/// An [`ExecDriver`] implementation that spawns the futures onto a Tokio runtime.
pub struct TokioDriver {
    pub handle: Handle,
    pub concurrency: usize,
}

impl ExecDriver for TokioDriver {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<Option<ArrayData>>>>,
    ) -> BoxStream<'static, VortexResult<ArrayData>> {
        let handle = self.handle.clone();

        stream
            .map(move |future| handle.spawn(future))
            .buffered(self.concurrency)
            .map(|result| match result {
                Ok(result) => result,
                Err(e) => Err(vortex_err!("Failed to join Tokio result {}", e)),
            })
            .filter_map(|r| ready(r.transpose()))
            .boxed()
    }
}
