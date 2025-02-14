use std::future::Future;

use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt};
use tokio::runtime::Handle;
use vortex_error::{VortexError, VortexResult};

use super::Spawn;

/// Tokio-based async task executor, runs task on the provided runtime.
#[derive(Clone)]
pub struct TokioExecutor(Handle);

impl TokioExecutor {
    pub fn new(handle: Handle) -> Self {
        Self(handle)
    }
}

#[async_trait::async_trait]
impl Spawn for TokioExecutor {
    fn spawn<F>(&self, f: F) -> VortexResult<BoxFuture<'static, VortexResult<F::Output>>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        Ok(self.0.spawn(f).map_err(VortexError::from).boxed())
    }
}
