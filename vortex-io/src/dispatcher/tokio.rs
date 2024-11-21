use std::future::Future;
use std::panic::resume_unwind;
use std::thread::JoinHandle;

use futures::channel::oneshot;
use tokio::task::{JoinHandle as TokioJoinHandle, LocalSet};
use vortex_error::{vortex_bail, vortex_panic, VortexResult};

use super::Dispatch;

trait TokioSpawn {
    fn spawn(self: Box<Self>) -> TokioJoinHandle<()>;
}

/// A [dispatcher][Dispatch] of IO operations that runs tasks on one of several
/// Tokio `current_thread` runtimes.
#[derive(Debug)]
pub(super) struct TokioDispatcher {
    submitter: flume::Sender<Box<dyn TokioSpawn + Send>>,
    threads: Vec<JoinHandle<()>>,
}

impl TokioDispatcher {
    pub fn new(num_threads: usize) -> Self {
        let (submitter, rx) = flume::unbounded();
        let threads: Vec<_> = (0..num_threads)
            .map(|tid| {
                let worker_thread =
                    std::thread::Builder::new().name(format!("tokio-dispatch-{tid}"));
                let rx: flume::Receiver<Box<dyn TokioSpawn + Send>> = rx.clone();

                worker_thread
                    .spawn(move || {
                        // Create a runtime-per-thread
                        let rt = tokio::runtime::Builder::new_current_thread()
                            // The dispatcher should not have any blocking work.
                            // Maybe in the future we can add this as a config param.
                            .max_blocking_threads(1)
                            .enable_all()
                            .build()
                            .unwrap_or_else(|e| {
                                vortex_panic!("TokioDispatcher new_current_thread build(): {e}")
                            });

                        rt.block_on(async move {
                            // Use a LocalSet so that all spawned tasks will run on the current thread. This allows
                            // spawning !Send futures.
                            LocalSet::new()
                                .run_until(async {
                                    while let Ok(task) = rx.recv_async().await {
                                        task.spawn();
                                    }
                                })
                                .await;
                        });
                    })
                    .unwrap_or_else(|e| vortex_panic!("TokioDispatcher worker thread spawn: {e}"))
            })
            .collect();

        Self { submitter, threads }
    }
}

/// Tasks that can be launched onto a runtime.
struct TokioTask<F, R> {
    task: F,
    result: oneshot::Sender<R>,
}

impl<F, Fut, R> TokioSpawn for TokioTask<F, R>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = R>,
    R: Send + 'static,
{
    fn spawn(self: Box<Self>) -> TokioJoinHandle<()> {
        let TokioTask { task, result } = *self;
        tokio::task::spawn_local(async move {
            let task_output = task().await;
            result.send(task_output).ok();
        })
    }
}

impl Dispatch for TokioDispatcher {
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<oneshot::Receiver<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        let task = TokioTask { result: tx, task };

        match self.submitter.send(Box::new(task)) {
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
            // NOTE: currently, panics inside any of the tasks will not propagate to the LocalSet's join handle,
            // see https://docs.rs/tokio/latest/tokio/task/struct.LocalSet.html#panics-1
            thread.join().unwrap_or_else(|err| resume_unwind(err));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::TokioDispatcher;
    use crate::dispatcher::Dispatch;
    use crate::{TokioFile, VortexReadAt};

    #[tokio::test]
    async fn test_tokio_dispatch_simple() {
        let dispatcher = TokioDispatcher::new(4);
        let mut tmpfile = NamedTempFile::new().unwrap();
        write!(tmpfile, "5678").unwrap();

        let rx = dispatcher
            .dispatch(|| async move {
                let file = TokioFile::open(tmpfile.path()).unwrap();

                file.read_byte_range(0, 4).await.unwrap()
            })
            .unwrap();

        assert_eq!(&rx.await.unwrap(), "5678".as_bytes());
    }
}
