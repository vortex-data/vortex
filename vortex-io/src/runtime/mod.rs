// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use handle::*;
use std::sync::Arc;
mod coalesce;
mod handle;
pub mod io;
pub mod multithread;
pub mod singlethread;
pub mod tokio;
pub mod worker;

use crate::runtime::coalesce::CoalescedRequest;
use crate::runtime::io::IoSource;
use crate::runtime::io::ReadCompletion;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_buffer::Alignment;
use vortex_error::VortexExpect;

/// A Vortex runtime provides an abstract way of scheduling mixed I/O and CPU workloads onto the
/// various threading models supported by Vortex.
///
/// The models we currently support are:
/// * Single-threaded: all work is driven on the current thread.
/// * Multi-threaded: work is driven on a pool of threads managed by Vortex.
/// * Worker Pool: work is driven on a pool of threads provided by the caller.
/// * Tokio: work is driven on a Tokio runtime provided by the caller.
///
/// Note: users will interact with the [`Handle`] API rather than the [`Runtime`] trait.
///
/// FIXME(ngates): these should really have handles that get dropped and cancel?
pub(crate) trait Runtime<'rt>: Send + Sync {
    /// Spawns a future to be executed on the runtime's scheduling context.
    ///
    /// The future will continue to be executed in the background and should pass results out via
    /// a one-shot channel if necessary.
    fn spawn_scheduling(&self, fut: BoxFuture<'rt, ()>);

    /// Spawns a CPU-bound task for execution on the runtime.
    fn spawn_cpu(&self, task: CpuTask);

    /// Passes a stream of I/O tasks to be executed on the runtime.
    fn spawn_io(&self, stream: BoxStream<'rt, IoTask>, concurrency: usize);
}

pub(crate) struct CpuTask {
    runnable: Box<dyn FnOnce() + Send + 'static>,
    // TODO(ngates): we may want worker affinity and other metadata in here?
    //  We may also just want to use an async task Runnable and accept that it's blocking?
}

impl CpuTask {
    pub(crate) fn run(self) {
        (self.runnable)()
    }
}

// NOTE(ngates): we may well want to make this an enum so we have better control over the common
//  cases of files and object store, we can always have a fallback to arbitrary futures.
pub struct IoTask {
    source: Arc<dyn IoSource>,
    request: IoRequest,
}

impl IoTask {
    pub fn new_single(source: Arc<dyn IoSource>, request: ReadRequest) -> Self {
        IoTask {
            source,
            request: IoRequest::Single(request),
        }
    }

    pub fn new_coalesced(source: Arc<dyn IoSource>, request: CoalescedRequest) -> Self {
        IoTask {
            source,
            request: IoRequest::Coalesced(request),
        }
    }
}

impl IoTask {
    /// Run this task as a `!Send` future.
    ///
    /// In some cases, this is more optimized than the `Send` version if the calling runtime
    /// supports it.
    pub async fn run_local(self) {
        if self.request.is_canceled() {
            // If the request has been cancelled by the time we come to execute it, just skip it.
            return;
        }

        match self.request {
            IoRequest::Single(req) => req.callback.complete(
                self.source
                    .read_local(req.offset, req.length, req.alignment)
                    .await,
            ),
            IoRequest::Coalesced(req) => {
                let result = self
                    .source
                    .read_local(
                        req.range.start,
                        usize::try_from(req.range.end - req.range.start)
                            .vortex_expect("too big for usize"),
                        req.alignment,
                    )
                    .await;
                req.resolve(result)
            }
        }
    }

    /// Run this task as a `Send` future.
    pub async fn run_send(self) {
        match self.request {
            IoRequest::Single(req) => req.callback.complete(
                self.source
                    .read_send(req.offset, req.length, req.alignment)
                    .await,
            ),
            IoRequest::Coalesced(req) => {
                let result = self
                    .source
                    .read_send(
                        req.range.start,
                        usize::try_from(req.range.end - req.range.start)
                            .vortex_expect("too big for usize"),
                        req.alignment,
                    )
                    .await;
                req.resolve(result)
            }
        }
    }
}

/// An I/O request encapsulates either a single read or a coalesced read in a way that allows us
/// to track cancellation and completion.
pub enum IoRequest {
    Single(ReadRequest),
    Coalesced(CoalescedRequest),
}

impl IoRequest {
    pub fn is_canceled(&self) -> bool {
        match self {
            IoRequest::Single(req) => req.callback.is_canceled(),
            IoRequest::Coalesced(req) => req.requests.iter().all(|r| r.callback.is_canceled()),
        }
    }
}

pub struct ReadRequest {
    pub offset: u64,
    pub length: usize,
    pub alignment: Alignment,
    pub callback: ReadCompletion,
}
