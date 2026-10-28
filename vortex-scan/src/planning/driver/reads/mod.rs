// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-execution read sharing, bounded by the earliest unfinished consumer row.

mod progress;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::future::poll_fn;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;
use std::thread;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::AbortHandle;
use futures::future::Abortable;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

pub use self::progress::RowClaim;

/// The last possible use of a read, expressed in the file's root row domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReadLifetime {
    /// No consumer at or beyond this exclusive row end can need the read.
    ThroughRow(u64),
    /// No tighter bound is known; keep the read until the file execution ends.
    File,
}

/// Conservative read lifetimes derived by the layout planner. Missing targets live to file end.
pub type ReadLifetimes = Arc<HashMap<IoTarget, ReadLifetime>>;

type SharedRead = Shared<BoxFuture<'static, SharedVortexResult<IoResult>>>;

/// Read sharing for one execution over one file, across all its workers and queued work.
///
/// Reserve the scan's rows with [`claim`](Self::claim), then move and split that claim with
/// the work. Queued work retains its claim before any worker takes it. Completed reads and
/// unconsumed hints expire when all claims have passed their row end. Dropping the last nonempty
/// claim clears even file-lifetime reads, independently of the service or prepared plan lifetime.
#[derive(Clone)]
pub struct FileReads {
    transport: Arc<dyn IoService>,
    lifetimes: ReadLifetimes,
    state: Arc<Mutex<ReadState>>,
}

#[derive(Default)]
struct ReadState {
    claims: BTreeMap<u64, usize>,
    entries: HashMap<IoTarget, ReadEntry>,
    expiry: BTreeMap<u64, Vec<IoTarget>>,
}

struct ReadEntry {
    transport: Arc<ReadTransport>,
    read: Option<SharedRead>,
}

/// One target's transport scope. Clearing it also releases unused announcements on eviction.
struct ReadTransport(Arc<dyn IoSource>);

impl Drop for ReadTransport {
    fn drop(&mut self) {
        self.0.clear();
    }
}

impl FileReads {
    /// Creates fresh sharing state for one file execution. Lifetimes may be reused across
    /// executions, but the service itself must not be shared with another execution or file.
    pub fn new(transport: Arc<dyn IoService>, lifetimes: ReadLifetimes) -> Self {
        Self {
            transport,
            lifetimes,
            state: Arc::default(),
        }
    }

    /// Reserves the initial rows before any work runs. Descendants receive splits or shares
    /// of this claim; they must not register fresh ranges behind the execution's watermark.
    pub fn claim(&self, rows: Range<u64>) -> RowClaim {
        RowClaim::new(self.clone(), rows)
    }

    fn request(&self, request: IoRequest) -> VortexResult<Option<SharedRead>> {
        let mut state = self.state.lock();
        vortex_ensure!(
            !state.claims.is_empty(),
            "file reads require a live row claim"
        );
        let ReadState {
            entries, expiry, ..
        } = &mut *state;
        let entry = match entries.entry(request.target) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let lifetime = self
                    .lifetimes
                    .get(&request.target)
                    .copied()
                    .unwrap_or(ReadLifetime::File);
                if let ReadLifetime::ThroughRow(end) = lifetime {
                    expiry.entry(end).or_default().push(request.target);
                }
                entry.insert(ReadEntry {
                    transport: Arc::new(ReadTransport(self.transport.session())),
                    read: None,
                })
            }
        };
        if request.intent == IoIntent::Announce {
            if entry.read.is_none() {
                entry.transport.0.submit(
                    IoOwnerId(0),
                    vec![IoRequest {
                        request: IoRequestId(0),
                        ..request
                    }],
                )?;
            }
            return Ok(None);
        }
        if let Some(read) = &entry.read {
            return Ok(Some(read.clone()));
        }
        // Prefetches become internal fetches: the coordinator owns their result, and only a
        // subscribing fetch gets a completion under its original owner and request identity.
        entry.transport.0.submit(
            IoOwnerId(0),
            vec![IoRequest {
                intent: IoIntent::Fetch,
                request: IoRequestId(0),
                ..request
            }],
        )?;
        let transport = Arc::clone(&entry.transport);
        let read = async move {
            let completion = poll_fn(|cx| transport.0.poll_completion(cx)).await?;
            vortex_ensure!(
                completion.owner == IoOwnerId(0) && completion.request == IoRequestId(0),
                "transport completed an unknown read"
            );
            let result = completion.result?;
            if !result.matches(&request.target) {
                return Err(vortex_err!(
                    "read of {:?} returned {}",
                    request.target,
                    result.kind()
                ));
            }
            Ok(result)
        }
        .map(|result: VortexResult<IoResult>| result.map_err(Arc::new))
        .boxed()
        .shared();
        entry.read = Some(read.clone());
        Ok(Some(read))
    }
}

impl IoService for FileReads {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(ReadSession {
            file: self.clone(),
            pending: Mutex::default(),
        })
    }
}

#[derive(Default)]
struct Pending {
    reads: FuturesUnordered<BoxFuture<'static, Option<Completion>>>,
    ready: VecDeque<Completion>,
    subscriptions: HashMap<(IoOwnerId, IoRequestId), AbortHandle>,
}

struct ReadSession {
    file: FileReads,
    pending: Mutex<Pending>,
}

impl ReadSession {
    fn next(&self, cx: &mut Context<'_>) -> Poll<Option<Completion>> {
        let mut pending = self.pending.lock();
        if let Some(completion) = pending.ready.pop_front() {
            return Poll::Ready(Some(completion));
        }
        loop {
            match pending.reads.poll_next_unpin(cx) {
                Poll::Ready(Some(Some(completion))) => {
                    pending
                        .subscriptions
                        .remove(&(completion.owner, completion.request));
                    return Poll::Ready(Some(completion));
                }
                Poll::Ready(Some(None)) => continue,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl IoSource for ReadSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut pending = self.pending.lock();
        let mut reads = Vec::with_capacity(batch.len());
        for request in batch {
            if let Some(read) = self.file.request(request)? {
                reads.push((request, read));
            }
        }
        for (request, mut read) in reads {
            let ready = read.poll_unpin(&mut Context::from_waker(Waker::noop()));
            if request.intent == IoIntent::Prefetch {
                if let Poll::Ready(Err(error)) = ready {
                    return Err(VortexError::from(error));
                }
                continue;
            }
            // Cached and in-memory reads need no asynchronous subscription allocation.
            if let Poll::Ready(result) = ready {
                pending.ready.push_back(Completion {
                    owner,
                    request: request.request,
                    result: result.map_err(VortexError::from),
                });
                continue;
            }
            let (abort, registration) = AbortHandle::new_pair();
            pending
                .subscriptions
                .insert((owner, request.request), abort);
            let completion = async move {
                Completion {
                    owner,
                    request: request.request,
                    result: read.await.map_err(VortexError::from),
                }
            };
            pending.reads.push(
                Abortable::new(completion, registration)
                    .map(Result::ok)
                    .boxed(),
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(match self.next(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(completion) => completion,
            Poll::Pending => None,
        })
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        match self.next(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!("waiting with no reads in flight"))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
        let mut cx = Context::from_waker(&waker);
        loop {
            match self.poll_completion(&mut cx) {
                Poll::Ready(completion) => return completion,
                Poll::Pending => thread::park(),
            }
        }
    }

    fn release(&self, owner: IoOwnerId) {
        let mut pending = self.pending.lock();
        pending.ready.retain(|completion| completion.owner != owner);
        pending.subscriptions.retain(|(subscriber, _), abort| {
            if *subscriber == owner {
                abort.abort();
                false
            } else {
                true
            }
        });
    }

    fn clear(&self) {
        *self.pending.lock() = Pending::default();
    }
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
