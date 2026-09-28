// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::mpsc;
use std::thread;

use futures::channel::oneshot;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::parallelism::get_available_parallelism;

/// Driver threads per available core. Drivers spend much of their time waiting for reads, so the
/// pool overlaps several per core.
const THREADS_PER_CORE: usize = 4;

type Job = Box<dyn FnOnce() + Send>;

/// Threads that run split drivers, which block while they wait for reads.
///
/// The drivers must not run on the runtime's blocking pool: the reads they wait for may need that
/// pool too, as local object-store reads do, and a pool full of waiting drivers never runs them.
/// These threads are used for nothing else, so a waiting driver never holds up a read.
struct DriverPool {
    jobs: mpsc::Sender<Job>,
}

static POOL: LazyLock<DriverPool> = LazyLock::new(|| {
    let (jobs, receiver) = mpsc::channel::<Job>();
    let receiver = Arc::new(Mutex::new(receiver));
    let threads = get_available_parallelism().unwrap_or(1) * THREADS_PER_CORE;
    for index in 0..threads {
        let receiver = Arc::clone(&receiver);
        drop(
            thread::Builder::new()
                .name(format!("vortex-scan-v2-driver-{index}"))
                .spawn(move || {
                    loop {
                        let job = receiver.lock().recv();
                        let Ok(job) = job else {
                            return;
                        };
                        job();
                    }
                }),
        );
    }
    DriverPool { jobs }
});

/// Runs `work` on a driver thread and returns its result. Panics are returned as errors.
pub(super) async fn run_on_driver_thread<R: Send + 'static>(
    work: impl FnOnce() -> VortexResult<R> + Send + 'static,
) -> VortexResult<R> {
    let (sender, receiver) = oneshot::channel();
    let job: Job = Box::new(move || {
        let result = catch_unwind(AssertUnwindSafe(work))
            .unwrap_or_else(|_| Err(vortex_err!("A split driver panicked")));
        // The scan may have been dropped while the split ran.
        drop(sender.send(result));
    });
    POOL.jobs
        .send(job)
        .map_err(|_| vortex_err!("The split driver pool has shut down"))?;
    receiver
        .await
        .map_err(|_| vortex_err!("A split driver stopped without a result"))?
}
