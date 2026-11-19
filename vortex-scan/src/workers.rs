// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Running a scan on threads that Vortex provides.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::SinkExt;
use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc;
use vortex_array::ArrayRef;
use vortex_array::expr::stats::Precision;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;

use crate::ScanPoolRef;
use crate::ScanRef;

/// The fewest pieces an ordered scan is cut into for each worker.
const ORDERED_PIECES_PER_WORKER: usize = 4;

/// The most bytes of source data an ordered piece covers.
///
/// Pieces behind the one being emitted run ahead and buffer their whole output, so the memory held
/// for ordering is about `workers` pieces. Cutting by size keeps that bound fixed however large
/// the scan is.
const ORDERED_PIECE_BYTES: u64 = 64 << 20;

/// Runs `scan` on `workers` tasks spawned onto `handle` and returns their output as one stream.
///
/// This is for callers that want Vortex to provide the threads. A host that has its own threads
/// should [`open`](crate::Scan::open) the scan and poll [`worker`](crate::ScanPool::worker)s
/// itself.
///
/// - An unordered scan opens one pool and runs `workers` workers on it, merging their output in
///   completion order.
/// - An ordered scan is [`split`](crate::Scan::split) into contiguous pieces. At most `workers`
///   pieces run at once, each read by one worker, and their outputs are emitted in piece order. A
///   scan that cannot be cut, such as one with a limit, runs on a single worker.
///
/// Splitting and opening read metadata only. Nothing is spawned until the stream is first polled.
pub fn execute_with_workers(
    scan: ScanRef,
    workers: NonZeroUsize,
    handle: &Handle,
) -> VortexResult<SendableArrayStream> {
    let dtype = scan.dtype().clone();
    if scan.ordered() {
        let pieces = ordered_piece_count(scan.byte_size(), workers);
        let pieces = scan.split(pieces)?;
        let stream = OrderedPieces {
            pending: pieces.into(),
            running: VecDeque::new(),
            workers: workers.get(),
            handle: handle.clone(),
        };
        return Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )));
    }

    let pool = scan.open()?;
    let stream = UnorderedWorkers::Idle {
        pool,
        workers: workers.get(),
        handle: handle.clone(),
    };
    Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
        dtype, stream,
    )))
}

/// How many pieces to cut an ordered scan of `byte_size` into for `workers` workers.
fn ordered_piece_count(byte_size: Precision<u64>, workers: NonZeroUsize) -> NonZeroUsize {
    let by_workers = workers.get().saturating_mul(ORDERED_PIECES_PER_WORKER);
    let by_size = byte_size
        .into_inner()
        .map_or(0, |bytes| bytes.div_ceil(ORDERED_PIECE_BYTES))
        .try_into()
        .unwrap_or(usize::MAX);
    NonZeroUsize::new(by_workers.max(by_size)).unwrap_or(workers)
}

/// Polls one worker of `pool` to completion, sending its arrays into `output`.
///
/// Stops after the first error, or when the receiver is gone.
async fn drive_worker(pool: ScanPoolRef, mut output: mpsc::Sender<VortexResult<ArrayRef>>) {
    let mut worker = pool.worker();
    while let Some(item) = worker.next().await {
        let is_err = item.is_err();
        if output.send(item).await.is_err() || is_err {
            break;
        }
    }
}

/// The workers of one unordered pool, spawned on first poll.
enum UnorderedWorkers {
    Idle {
        pool: ScanPoolRef,
        workers: usize,
        handle: Handle,
    },
    Running {
        output: mpsc::Receiver<VortexResult<ArrayRef>>,
        /// Dropping the stream drops these tasks, which cancels the workers.
        _tasks: Vec<Task<()>>,
    },
    Done,
}

impl Stream for UnorderedWorkers {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        loop {
            match this {
                UnorderedWorkers::Idle {
                    pool,
                    workers,
                    handle,
                } => {
                    let (sender, output) = mpsc::channel(*workers);
                    let tasks = (0..*workers)
                        .map(|_| handle.spawn(drive_worker(Arc::clone(pool), sender.clone())))
                        .collect();
                    *this = UnorderedWorkers::Running {
                        output,
                        _tasks: tasks,
                    };
                }
                UnorderedWorkers::Running { output, .. } => {
                    let item = ready!(output.poll_next_unpin(cx));
                    if item.as_ref().is_none_or(Result::is_err) {
                        // End after the first error: dropping the tasks cancels the other workers.
                        *this = UnorderedWorkers::Done;
                    }
                    return Poll::Ready(item);
                }
                UnorderedWorkers::Done => return Poll::Ready(None),
            }
        }
    }
}

/// One piece of an ordered scan, read by one spawned worker into its own buffer.
struct RunningPiece {
    output: mpsc::UnboundedReceiver<VortexResult<ArrayRef>>,
    _task: Task<()>,
}

/// The contiguous pieces of an ordered scan, emitted in piece order with at most `workers`
/// pieces running.
struct OrderedPieces {
    pending: VecDeque<ScanRef>,
    running: VecDeque<RunningPiece>,
    workers: usize,
    handle: Handle,
}

impl OrderedPieces {
    fn start_pieces(&mut self) -> VortexResult<()> {
        while self.running.len() < self.workers
            && let Some(piece) = self.pending.pop_front()
        {
            let pool = piece.open()?;
            let (sender, output) = mpsc::unbounded();
            let task = self.handle.spawn(async move {
                let mut worker = pool.worker();
                while let Some(item) = worker.next().await {
                    let is_err = item.is_err();
                    if sender.unbounded_send(item).is_err() || is_err {
                        break;
                    }
                }
            });
            self.running.push_back(RunningPiece {
                output,
                _task: task,
            });
        }
        Ok(())
    }
}

impl Stream for OrderedPieces {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        loop {
            if let Err(err) = this.start_pieces() {
                this.pending.clear();
                this.running.clear();
                return Poll::Ready(Some(Err(err)));
            }
            let Some(front) = this.running.front_mut() else {
                return Poll::Ready(None);
            };
            match ready!(front.output.poll_next_unpin(cx)) {
                Some(Ok(array)) => return Poll::Ready(Some(Ok(array))),
                Some(Err(err)) => {
                    this.pending.clear();
                    this.running.clear();
                    return Poll::Ready(Some(Err(err)));
                }
                None => {
                    this.running.pop_front();
                }
            }
        }
    }
}
