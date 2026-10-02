// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A blocking byte source for exercising the protocol with real footer reads.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use crate::VortexReadAt;
use crate::runtime::BlockingRuntime;

use super::Completion;
use super::IoBatch;
use super::IoIntent;
use super::IoRequest;
use super::IoResult;
use super::IoSource;
use super::IoTarget;
use super::IoOwnerId;

/// A simple source that registers batches and performs one fetch per `wait()`.
/// Optional hints are declined. Shared reads, prefetching, and priority scheduling
/// belong to a later IO service implementation.
pub struct ReadAtIoSource<R: BlockingRuntime> {
    read: Arc<dyn VortexReadAt>,
    runtime: Arc<R>,
    queued: Mutex<VecDeque<(IoOwnerId, IoRequest)>>,
}

impl<R: BlockingRuntime> ReadAtIoSource<R> {
    /// Creates a source that drives its reads on `runtime`.
    pub fn new(read: Arc<dyn VortexReadAt>, runtime: Arc<R>) -> Self {
        Self {
            read,
            runtime,
            queued: Mutex::default(),
        }
    }
}

impl<R: BlockingRuntime + Send + Sync> IoSource for ReadAtIoSource<R> {
    fn submit(&self, work: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        for request in &batch {
            if let IoTarget::Range { offset, len } = &request.target
                && offset.checked_add(*len as u64).is_none()
            {
                vortex_bail!("ReadAtIoSource: range overflow for {:?}", request.request);
            }
        }
        self.queued.lock().extend(
            batch
                .into_iter()
                .filter(|request| request.intent == IoIntent::Fetch)
                .map(|request| (work, request)),
        );
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn wait(&self) -> VortexResult<Completion> {
        let (work, request) = self
            .queued
            .lock()
            .pop_front()
            .ok_or_else(|| vortex_err!("ReadAtIoSource: wait with nothing outstanding"))?;
        let result = match request.target {
            IoTarget::Size => self.runtime.block_on(self.read.size()).map(IoResult::Size),
            IoTarget::Range { offset, len } => self
                .runtime
                .block_on(self.read.read_at(offset, len, Alignment::none()))
                .map(IoResult::Bytes),
        };
        Ok(Completion {
            owner: work,
            request: request.request,
            result,
        })
    }

    fn release(&self, work: IoOwnerId) {
        self.queued.lock().retain(|(owner, _)| *owner != work);
    }

    fn clear(&self) {
        self.queued.lock().clear();
    }
}

#[cfg(test)]
mod tests;
