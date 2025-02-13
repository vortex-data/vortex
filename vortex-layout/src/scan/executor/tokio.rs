use std::future::Future;

use futures::channel::oneshot;
use tokio::runtime::Handle;

use super::{JoinHandle, Spawn};

#[derive(Clone)]
pub struct TokioExecutor(Handle);

impl TokioExecutor {
    pub fn new(handle: Handle) -> Self {
        Self(handle)
    }
}

#[async_trait::async_trait]
impl Spawn for TokioExecutor {
    fn spawn<F>(&self, f: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        self.0.spawn(async move {
            let v = f.await;
            _ = tx.send(v);
        });

        JoinHandle { inner: rx }
    }
}
