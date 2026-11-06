// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A non-blocking IO service over any [`VortexReadAt`].

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::task::noop_waker_ref;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::Completion;
use super::IoBatch;
use super::IoIntent;
use super::IoOwnerId;
use super::IoResult;
use super::IoService;
use super::IoSource;
use super::IoTarget;
use crate::VortexReadAt;

/// Serves sizes and byte ranges of one [`VortexReadAt`], such as a file of a table, to planners
/// that need its raw bytes, like a footer read before the file's segments are known.
///
/// Every session starts each fetch's read as it is submitted and hands completions back as they
/// finish, without blocking. Optional hints are declined.
pub struct ReadAtIoService {
    read: Arc<dyn VortexReadAt>,
}

impl ReadAtIoService {
    /// A service reading from `read`.
    pub fn new(read: Arc<dyn VortexReadAt>) -> Self {
        Self { read }
    }
}

impl IoService for ReadAtIoService {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(ReadAtSession {
            read: Arc::clone(&self.read),
            reads: Mutex::default(),
            ready: Mutex::default(),
            released: Mutex::default(),
        })
    }
}

struct ReadAtSession {
    read: Arc<dyn VortexReadAt>,
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Completion>>>,
    ready: Mutex<VecDeque<Completion>>,
    /// Owners that retired with reads in flight, whose completions are dropped.
    released: Mutex<HashSet<IoOwnerId>>,
}

impl ReadAtSession {
    /// Polls in-flight reads with `cx`, returning the first completion of a live owner.
    fn poll_reads(&self, cx: &mut Context<'_>) -> Poll<Option<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Poll::Ready(Some(completion));
        }
        let mut reads = self.reads.lock();
        loop {
            match reads.poll_next_unpin(cx) {
                Poll::Ready(Some(completion)) => {
                    if self.released.lock().contains(&completion.owner) {
                        continue;
                    }
                    return Poll::Ready(Some(completion));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl IoSource for ReadAtSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        for request in batch {
            if request.intent != IoIntent::Fetch {
                continue;
            }
            let id = request.request;
            let read = match request.target {
                IoTarget::Size => self.read.size().map(|r| r.map(IoResult::Size)).boxed(),
                IoTarget::Range {
                    offset,
                    len,
                    alignment,
                } => {
                    if offset.checked_add(len as u64).is_none() {
                        vortex_bail!("ReadAtIoService: range overflow for {id:?}");
                    }
                    self.read
                        .read_at(offset, len, alignment)
                        .map(|r| r.map(IoResult::Bytes))
                        .boxed()
                }
            };
            reads.push(
                read.map(move |result| Completion {
                    owner,
                    request: id,
                    result,
                })
                .boxed(),
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        let mut cx = Context::from_waker(noop_waker_ref());
        Ok(match self.poll_reads(&mut cx) {
            Poll::Ready(completion) => completion,
            Poll::Pending => None,
        })
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        self.poll_reads(cx).map(|completion| {
            completion.ok_or_else(|| vortex_err!("ReadAtIoService: wait with nothing outstanding"))
        })
    }

    fn wait(&self) -> VortexResult<Completion> {
        futures::executor::block_on(futures::future::poll_fn(|cx| self.poll_completion(cx)))
    }

    fn release(&self, owner: IoOwnerId) {
        self.ready.lock().retain(|c| c.owner != owner);
        if !self.reads.lock().is_empty() {
            self.released.lock().insert(owner);
        }
    }

    fn clear(&self) {
        self.ready.lock().clear();
        *self.reads.lock() = FuturesUnordered::new();
        self.released.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::request::IoRequest;
    use crate::request::IoRequestId;

    fn fetch(id: u32, target: IoTarget) -> IoRequest {
        IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(id),
            target,
        }
    }

    #[test]
    fn serves_sizes_and_ranges() -> VortexResult<()> {
        let bytes = ByteBuffer::from((0u8..100).collect::<Vec<_>>());
        let service = ReadAtIoService::new(Arc::new(bytes));
        let session = service.session();
        session.submit(
            IoOwnerId(1),
            vec![fetch(0, IoTarget::Size), fetch(1, IoTarget::range(90, 4))],
        )?;
        let mut size = None;
        let mut range = None;
        for _ in 0..2 {
            let completion = session.wait()?;
            assert_eq!(completion.owner, IoOwnerId(1));
            match completion.result? {
                IoResult::Size(s) => size = Some(s),
                IoResult::Bytes(b) => range = Some(b.try_to_host_sync()?.to_vec()),
            }
        }
        assert_eq!(size, Some(100));
        assert_eq!(range, Some(vec![90, 91, 92, 93]));
        assert!(session.poll()?.is_none());
        Ok(())
    }
}
