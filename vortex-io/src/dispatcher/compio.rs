use std::future::Future;
use std::panic::resume_unwind;
use std::thread::JoinHandle;

use compio::runtime::{JoinHandle as CompioJoinHandle, Runtime, RuntimeBuilder};
use futures::channel::oneshot;
use vortex_error::{vortex_bail, vortex_panic, VortexResult};

use super::Dispatch;

trait CompioSpawn {
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()>;
}

struct CompioTask<F, R> {
    task: F,
    result: oneshot::Sender<R>,
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
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<oneshot::Receiver<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let compio_task = Box::new(CompioTask { task, result: tx });
        match self.submitter.send(compio_task) {
            Ok(()) => Ok(rx),
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
}
