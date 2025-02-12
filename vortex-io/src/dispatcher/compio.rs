use std::future::Future;
use std::panic::resume_unwind;
use std::thread::JoinHandle;

use compio::runtime::{JoinHandle as CompioJoinHandle, Runtime, RuntimeBuilder};
use futures::channel::{mpsc, oneshot};
use futures::{SinkExt, Stream, StreamExt};
use vortex_error::{vortex_bail, vortex_panic, VortexResult};

use super::{Dispatch, JoinHandle as VortexJoinHandle, StreamHandle};

trait CompioSpawn {
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()>;
}

struct CompioTask<S, R> {
    task: S,
    result: oneshot::Sender<R>,
}

struct CompioStreamTask<S, R> {
    stream: S,
    sender: mpsc::Sender<R>,
}

impl<F, Fut, R> CompioSpawn for CompioTask<F, R>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = R>,
    R: Send + 'static,
{
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()> {
        let CompioTask { task, result } = *self;
        Runtime::with_current(|rt| {
            rt.spawn(async move {
                let task_output = task().await;
                result.send(task_output).ok();
            })
        })
    }
}

impl<S, R> CompioSpawn for CompioStreamTask<S, R>
where
    S: Stream<Item = R> + Unpin + 'static,
    R: 'static,
{
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()> {
        let Self {
            mut stream,
            mut sender,
        } = *self;

        Runtime::with_current(|rt| {
            rt.spawn(async move {
                while let Some(v) = stream.next().await {
                    let r = sender.send(v).await;

                    if r.is_err() {
                        return;
                    }
                }
            })
        })
    }
}

#[derive(Debug)]
pub(super) struct CompioDispatcher {
    submitter: flume::Sender<Box<dyn CompioSpawn + Send>>,
    threads: Vec<JoinHandle<()>>,
}

impl CompioDispatcher {
    pub fn new(num_threads: usize) -> Self {
        let (submitter, rx) = flume::unbounded();
        let threads: Vec<_> = (0..num_threads)
            .map(|tid| {
                let worker_thread = std::thread::Builder::new();
                let worker_thread = worker_thread.name(format!("compio-dispatch-{tid}"));
                let rx: flume::Receiver<Box<dyn CompioSpawn + Send>> = rx.clone();

                worker_thread
                    .spawn(move || {
                        // Create a runtime-per-thread
                        let rt = RuntimeBuilder::new().build().unwrap_or_else(|e| {
                            vortex_panic!("CompioDispatcher RuntimeBuilder build(): {e}")
                        });

                        rt.block_on(async move {
                            while let Ok(task) = rx.recv_async().await {
                                task.spawn().detach();
                            }
                        });
                    })
                    .unwrap_or_else(|e| vortex_panic!("CompioDispatcher worker thread spawn: {e}"))
            })
            .collect();

        Self { submitter, threads }
    }
}

impl Dispatch for CompioDispatcher {
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<VortexJoinHandle<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let compio_task = Box::new(CompioTask { task, result: tx });
        match self.submitter.send(compio_task) {
            Ok(()) => Ok(VortexJoinHandle(rx)),
            Err(err) => vortex_bail!("Dispatcher error spawning task: {err}"),
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        // drop the submitter.
        //
        // Each worker thread will receive an `Err(Canceled)`
        drop(self.submitter);
        for thread in self.threads {
            // Propagate any panics from the worker threads.
            thread.join().unwrap_or_else(|err| resume_unwind(err));
        }

        Ok(())
    }

    fn drive_stream<S, T, E>(&self, stream: S) -> VortexResult<StreamHandle<Result<T, E>>>
    where
        T: Send + 'static,
        E: Send + 'static,
        S: Stream<Item = Result<T, E>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(1024);
        let stream = Box::pin(stream);
        let stream_task = Box::new(CompioStreamTask { stream, sender: tx });

        match self.submitter.send(stream_task) {
            Ok(()) => Ok(StreamHandle(rx)),
            Err(err) => vortex_bail!("Dispatcher error spawning task: {err}"),
        }
    }
}
