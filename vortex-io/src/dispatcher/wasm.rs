//! IoDispatcher that functions in WASM.

use std::future::Future;

use futures::channel::{mpsc, oneshot};
use futures::{SinkExt, Stream, StreamExt, TryStreamExt};
use vortex_error::{vortex_panic, VortexResult};
use wasm_bindgen_futures::wasm_bindgen::__rt::Start;

use super::{Dispatch, JoinHandle as VortexJoinHandle};

/// `Dispatch`able type that is available when running Vortex in the browser or other WASM env.
#[derive(Debug, Clone)]
pub struct WasmDispatcher;

impl WasmDispatcher {
    pub fn new() -> Self {
        WasmDispatcher
    }
}

impl Dispatch for WasmDispatcher {
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<VortexJoinHandle<R>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        wasm_bindgen_futures::spawn_local(async move {
            let result = task().await;
            tx.send(result)
                // NOTE: We don't know if the err is Debug
                .unwrap_or_else(|_err| vortex_panic!("WasmDispatcher: task submit failed"));
        })
        .start();

        Ok(VortexJoinHandle(rx))
    }

    fn drive_stream<S, T, E>(
        &self,
        stream: S,
    ) -> VortexResult<impl Stream<Item = Result<T, E>> + Send + 'static>
    where
        T: Send + 'static,
        E: Send + 'static,
        S: Stream<Item = Result<T, E>> + Send + 'static,
    {
        let (mut tx, rx) = mpsc::channel(1024);
        let mut stream = Box::pin(stream);

        wasm_bindgen_futures::spawn_local(async move {
            while let Some(v) = stream.next().await {
                let r = tx.send(v).await;
                if r.is_err() {
                    return;
                }
            }
        })
        .start();

        Ok(rx.into_stream())
    }

    fn shutdown(self) -> VortexResult<()> {
        Ok(())
    }
}
