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
    }
}
