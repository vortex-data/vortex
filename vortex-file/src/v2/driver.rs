use std::sync::Arc;

use futures::channel::oneshot;
use futures_executor::block_on;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use rayon::ThreadPool;
use vortex_array::ArrayData;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_layout::segments::AsyncSegmentReader;

pub trait Driver {
    /// Returns an [`AsyncSegmentReader`] used to construct a [`vortex_layout::LayoutReader`].
    fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static>;

    /// Spawn an evaluation task.
    /// The tasks will await on calls to the [`AsyncSegmentReader`] returned by [`Driver::reader`].
    fn spawn(
        &self,
        f: Box<dyn FnOnce() -> BoxFuture<VortexResult<ArrayData>>>,
    ) -> BoxFuture<VortexResult<ArrayData>> {
        f()
    }
}

pub struct ThreadPoolDriver {
    thread_pool: ThreadPool,
}

impl Driver for ThreadPoolDriver {
    fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static> {
        todo!()
    }

    fn spawn(
        &self,
        f: Box<dyn FnOnce() -> BoxFuture<VortexResult<ArrayData>>>,
    ) -> BoxFuture<VortexResult<ArrayData>> {
        let (send, recv) = oneshot::channel();

        // Launch the scan task onto the thread pool.
        self.thread_pool.spawn_fifo(move || {
            // Post the result back to the main thread
            send.send(block_on(f()))
                .map_err(|_| vortex_err!("send failed, recv dropped"))
                .vortex_expect("send_failed, recv dropped");
        });

        async move {
            recv.await
                .map_err(|_| vortex_err!("recv failed, send dropped"))
                .vortex_expect("recv failed, send dropped")
        }
        .boxed()
    }
}
