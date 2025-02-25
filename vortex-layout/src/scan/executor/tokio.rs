use std::future::Future;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt};
use tokio::runtime::Handle;
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use super::Executor;

/// Tokio-based async task executor, runs task on the provided runtime.
#[derive(Clone)]
pub struct TokioExecutor {
    inner: Arc<Inner>,
}

struct Inner {
    handle: Handle,
    // We use a joinset here so when the executor is dropped, it'll abort all running tasks
    join_set: Mutex<JoinSet<()>>,
}

impl TokioExecutor {
    pub fn new(handle: Handle) -> Self {
        let inner = Inner {
            handle,
            join_set: Mutex::new(JoinSet::new()),
        };

        Self {
            inner: Arc::new(inner),
        }
    }
}

#[async_trait::async_trait]
impl Executor for TokioExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let f = async move {
            let r = f.await;
            _ = tx.send(r);
        };

        self.inner
            .join_set
            .lock()
            .vortex_expect("poisoned lock")
            .spawn_on(f, &self.inner.handle);

        rx.map_err(|e| vortex_err!("Task sender dropped: {e}"))
            .boxed()
    }
}
