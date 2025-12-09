// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use tokio::runtime::Builder;
use tokio::runtime::Runtime;

type BoxedTask = Box<dyn FnOnce(&Runtime) + Send>;

/// A tokio runtime running on a separate thread.
///
/// This allows blocking code to dispatch async work to a dedicated runtime thread
/// and wait for results without blocking the runtime itself.
pub struct RemoteRuntime {
    sender: mpsc::Sender<BoxedTask>,
    handle: Option<JoinHandle<()>>,
}

impl RemoteRuntime {
    /// Creates a new RemoteRuntime with a tokio runtime on a dedicated thread.
    pub fn new(threads: Option<usize>) -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<BoxedTask>();

        let handle = std::thread::Builder::new()
            .name("remote-runtime".to_string())
            .spawn(move || {
                let rt = new_tokio_runtime(threads).expect("Failed to create tokio runtime");
                while let Ok(task) = receiver.recv() {
                    task(&rt);
                }
            })
            .context("Failed to spawn runtime thread")?;

        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    /// Executes an async future on the remote runtime and blocks until completion.
    ///
    /// Returns the result of the future.
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();

        self.sender
            .send(Box::new(move |rt| {
                let result = rt.block_on(future);
                let _ = tx.send(result);
            }))
            .expect("Runtime thread has shut down");

        rx.recv().expect("Runtime thread dropped the result")
    }
}

impl Drop for RemoteRuntime {
    fn drop(&mut self) {
        // Drop the sender to signal the thread to exit
        drop(std::mem::replace(&mut self.sender, mpsc::channel().0));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Creates a Tokio runtime based on the provided thread count configuration.
///
/// # Arguments
///
/// * `threads` - Optional number of threads to use:
///   * `None` - Uses Tokio's default multi-thread runtime
///   * `Some(0)` - Returns an error, as 0 threads is invalid
///   * `Some(1)` - Creates a single-threaded runtime
///   * `Some(n)` - Creates a multi-threaded runtime with `n` worker threads
///
/// # Returns
///
/// A configured Tokio runtime
///
/// # Errors
///
/// Returns an error if `threads` is `Some(0)` or if runtime creation fails
pub fn new_tokio_runtime(threads: Option<usize>) -> Result<Runtime> {
    match threads {
        Some(0) => bail!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .context("Failed building the Runtime")
}
