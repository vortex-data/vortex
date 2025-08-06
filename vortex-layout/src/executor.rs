// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::FutureExt;
use vortex_error::{vortex_err, ResultExt, VortexResult};

pub trait TaskExecutor: 'static + Send + Sync {
    fn do_spawn(
        &self,
        fut: BoxFuture<'static, VortexResult<()>>,
    ) -> BoxFuture<'static, VortexResult<()>>;
}

impl<T: TaskExecutor> TaskExecutor for Arc<T> {
    fn do_spawn(
        &self,
        fut: BoxFuture<'static, VortexResult<()>>,
    ) -> BoxFuture<'static, VortexResult<()>> {
        self.as_ref().do_spawn(fut)
    }
}

pub trait TaskExecutorExt: TaskExecutor {
    fn spawn<T: 'static + Send>(
        &self,
        fut: BoxFuture<'static, VortexResult<T>>,
    ) -> BoxFuture<'static, VortexResult<T>>;
}

impl<E: TaskExecutor + ?Sized> TaskExecutorExt for E {
    fn spawn<T: 'static + Send>(
        &self,
        fut: BoxFuture<'static, VortexResult<T>>,
    ) -> BoxFuture<'static, VortexResult<T>> {
        let (send, recv) = oneshot::channel::<VortexResult<T>>();
        let fut = self.do_spawn(
            async move {
                let result = fut.await;
                send.send(result)
                    .map_err(|_| vortex_err!("Failed to send result"))
            }
            .boxed(),
        );

        Box::pin(async move {
            fut.await?;
            recv.await
                .map_err(|canceled| vortex_err!("Spawned task canceled {}", canceled))
                .unnest()
        })
    }
}

#[cfg(feature = "tokio")]
impl TaskExecutor for tokio::runtime::Handle {
    fn do_spawn(
        &self,
        f: BoxFuture<'static, VortexResult<()>>,
    ) -> BoxFuture<'static, VortexResult<()>> {
        use futures::TryFutureExt;
        use tracing::Instrument;

        tokio::runtime::Handle::spawn(self, f.in_current_span())
            .map_err(vortex_error::VortexError::from)
            .map(|result| result.unnest())
            .boxed()
    }
}

struct LocalExecutor;

/// Returns a handle to an executor that will "spawn" tasks onto the current runtime.
pub fn local_task_executor() -> Arc<dyn TaskExecutor> {
    Arc::new(LocalExecutor)
}

impl TaskExecutor for LocalExecutor {
    fn do_spawn(
        &self,
        fut: BoxFuture<'static, VortexResult<()>>,
    ) -> BoxFuture<'static, VortexResult<()>> {
        fut
    }
}
