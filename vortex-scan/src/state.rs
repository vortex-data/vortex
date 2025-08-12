// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::tasks::TaskContext;
use async_stream::try_stream;
use crossbeam_channel::{Receiver, Sender, select_biased};
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt, TryStreamExt};
use parking_lot::Mutex;
use std::iter;
use std::ops::Range;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;
use takeaway::{Task, Worker};
use vortex_array::ArrayRef;
use vortex_array::stream::{ArrayStreamAdapter, ArrayStreamExt, SendableArrayStream};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::SegmentId;

pub struct Scan2 {}

impl Scan2 {
    pub fn try_new(
        ranges: Vec<Range<u64>>,
        ctx: TaskContext<ArrayRef>,
    ) -> Vec<SendableArrayStream> {
        // Create a futures unordered for spawned I/O requests.
        let mut io_requests = FuturesUnordered::new().into_stream().boxed();

        // Create a future that drives the I/O request stream.
        let io_fut = async move {
            while let Some(req) = io_requests.next().await {
                println!("Got I/O request: {:?}", req);
            }
            Ok(())
        }
            .boxed();

        let config = takeaway::Config::default().with_oneshot(true);
        let num_workers = config.num_workers();

        let cpu_queue = Arc::new(config.build::<WorkItem>());

        (0..num_workers.get())
            .into_iter()
            .map(move |worker_idx| {
                let queue = cpu_queue.clone();
                let mut worker = Worker::new(&queue, worker_idx);

                let cpu_stream = try_stream! {
                    while let Some(task) = worker.next().await {
                        match task {
                            WorkItem::FilterEvaluation { split_idx, conjunct_idx } => {
                                println!("Filter evaluation task: split_idx: {}, conjunct_idx: {}", split_idx, conjunct_idx);
                            },
                            WorkItem::ProjectEvaluation { split_idx } => {
                                println!("Project evaluation task: split_idx: {}", split_idx);
                            },
                        }
                        yield Err(vortex_err!("Failed"))
                    }
                };

                stream.select()

                ArrayStreamExt::boxed(ArrayStreamAdapter::new(DType::Null, stream))
            })
            .collect()
    }
}

enum WorkItem {
    FilterEvaluation {
        split_idx: usize,
        conjunct_idx: usize,
    },
    ProjectEvaluation {
        split_idx: usize,
    },
}

impl Task for WorkItem {
    type Priority = ();

    fn priority(&self) -> Self::Priority {
        // TODO(ngates): experiment with different conjunct priorities by selectivity?
        ()
    }
}

#[derive(Debug)]
enum IoItem {}

#[derive(Debug)]
enum CPUItem {
    InitSplit(usize),
    SplitFilter { conjunct_idx: usize },
    SplitProject,
    IoComplete { segment_id: SegmentId },
}

/// Each scan worker has its own local state and a handle to the global state.
pub struct ScanWorker {
    global: Arc<GlobalState>,

    // Local handle to the shared I/O queue.
    io_rx: Receiver<IoItem>,
    // Notification channel that a new CPU task is available.
    cpu_rx: Receiver<()>,

    cpu_worker: Worker<CPUItem>,
}

impl ScanWorker {
    /// Handle a single I/O message.
    fn handle_io(&mut self, msg: Result<IoItem, crossbeam_channel::RecvError>) -> VortexResult<()> {
        let msg = msg.map_err(|e| vortex_err!("I/O channel error: {e}"))?;
        println!("Handling I/O message: {:?}", msg);
        Ok(())
    }

    /// Handle a single CPU task.
    fn handle_cpu(&mut self, task: CPUItem) -> VortexResult<()> {
        println!("Handling CPU task: {:?}", task);
        Ok(())
    }

    /// Find the next CPU task.
    fn find_task(&mut self) -> Option<CPUItem> {
        // Pop a task from the local queue, if not empty.
        self.cpu_worker.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.global
                    .cpu_injector
                    .steal_batch_and_pop(&self.cpu_worker)
                    // Or try stealing a task from one of the other threads.
                    .or_else(|| {
                        self.global
                            .cpu_stealers
                            .read()
                            .iter()
                            .map(|s| s.steal())
                            .collect()
                    })
            })
                // Loop while no task was stolen and any steal operation needs to be retried.
                .find(|s| !s.is_retry())
                // Extract the stolen task, if there is one.
                .and_then(|s| s.success())
        })
    }
}

impl Iterator for ScanWorker {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // We bias towards I/O work, so that we can process I/O work and get going on more.
            select_biased! {
                recv(self.io_rx) -> msg => {
                    // We definitely have new I/O work to process.
                    if let Err(e) = self.handle_io(msg) {
                        return Some(Err(e));
                    }
                    continue;
                },
                recv(self.cpu_rx) -> _ => {
                    // We may have new CPU work to do, so break out of the select.
                },
            }

            // Try to pull CPU work from the work-stealing queue.
            if let Some(task) = self.find_task() {
                if let Err(e) = self.handle_cpu(task) {
                    return Some(Err(e));
                }
            }
        }
    }
}

// Custom waker that sends notifications to a channel
struct ThreadPoolWaker {
    sender: Sender<()>,
}

impl Wake for ThreadPoolWaker {
    fn wake(self: Arc<Self>) {
        // Non-blocking send, ignore if the channel is full
        let _ = self.sender.try_send(());
    }
}

// Wrapper for your shared future
struct SharedIoFuture {
    // The actual future, pinned and behind a mutex
    future: Arc<Mutex<BoxFuture<'static, VortexResult<()>>>>,
    // Channel for wake notifications
    wake_rx: Receiver<()>,
    wake_tx: Sender<()>,
}

impl SharedIoFuture {
    fn new<F>(future: F) -> Self
    where
        F: Future<Output=VortexResult<()>> + Send + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(1); // Bounded channel with capacity 1

        SharedIoFuture {
            future: Arc::new(Mutex::new(Box::pin(future))),
            wake_rx: rx,
            wake_tx: tx,
        }
    }

    // Try to poll the future, returns None if locked by another thread
    fn try_poll(&self) -> Option<Poll<VortexResult<()>>> {
        // Try to acquire lock without blocking
        let mut future_guard = self.future.try_lock()?;

        // Create waker for this poll attempt
        let waker = Waker::from(Arc::new(ThreadPoolWaker {
            sender: self.wake_tx.clone(),
        }));
        let mut context = Context::from_waker(&waker);

        // Poll the future
        Some(future_guard.as_mut().poll(&mut context))
    }

    // Wait for a wake notification with timeout
    fn wait_for_wake(&self, timeout: Duration) -> bool {
        self.wake_rx.recv_timeout(timeout).is_ok()
    }
}
