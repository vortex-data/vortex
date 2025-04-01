use std::future::Future;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use tokio::runtime::Handle;
use vortex_error::VortexExpect;

use super::Executor;

/// Tokio-based async task executor, runs task on the provided runtime.
#[derive(Clone)]
pub struct TokioExecutor {
    inner: Arc<Inner>,
}

struct Inner {
    handle: Handle,
}

impl TokioExecutor {
    pub fn new(handle: Handle) -> Self {
        let inner = Inner { handle };
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[async_trait::async_trait]
impl Executor for TokioExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        let handle = self.inner.handle.clone();
        async move { handle.spawn(f).await.vortex_expect("Failed to join task") }.boxed()
        //
        // let (tx, rx) = oneshot::channel();
        // let f = async move {
        //     let r = f.await;
        //     if let Err(_) = tx.send(r) {
        //         log::debug!("Dispatcher task receiver dropped before completion");
        //     }
        // };
        //
        //
        // // self.inner
        // //     .join_set
        // //     .lock()
        // //     .vortex_expect("poisoned lock")
        // //     .spawn_on(f, &self.inner.handle);
        //
        // async move {
        //     rx.await
        //         .map_err(|e| vortex_err!("Task sender dropped before completion: {}", e))
        //         .vortex_expect("Task sender dropped before completion")
        // }
        // .boxed()
    }
}
