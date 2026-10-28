// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use super::ReadAt2Factory;
use super::ReadCompletion;
use super::ReadId;
use super::ReadRequest;
use super::ReadResult;
use super::ReadTarget;
use super::VortexReadAt2;
use crate::request::Completion;
use crate::request::IoBatch;
use crate::request::IoIntent;
use crate::request::IoOwnerId;
use crate::request::IoRequestId;
use crate::request::IoResult;
use crate::request::IoService;
use crate::request::IoSource;
use crate::request::IoTarget;

/// Serves planner requests directly through completion-based positional reads.
///
/// Each split gets its own scope. Optional prefetch and announcement hints are declined: this
/// service retains no segment cache. The reader backend controls routing and coalescing.
pub struct ReadAt2IoService {
    factory: ReadAt2Factory,
}

impl ReadAt2IoService {
    /// Creates a service whose reader scopes are constructed on their driving worker.
    pub fn new(factory: ReadAt2Factory) -> Self {
        Self { factory }
    }
}

impl IoService for ReadAt2IoService {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(Source { reader: (self.factory)(), state: Mutex::default() })
    }
}

struct Source {
    reader: VortexResult<Arc<dyn VortexReadAt2>>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    next: u64,
    pending: HashMap<ReadId, (IoOwnerId, IoRequestId)>,
}

impl Source {
    fn reader(&self) -> VortexResult<&dyn VortexReadAt2> {
        self.reader.as_deref().map_err(|error| vortex_err!("Opening read scope: {error}"))
    }

    fn deliver(&self, completion: ReadCompletion) -> VortexResult<Option<Completion>> {
        let Some((owner, request)) = self.state.lock().pending.remove(&completion.id) else {
            return Ok(None);
        };
        let result = completion.result.and_then(|result| match result {
            ReadResult::Size(size) => Ok(IoResult::Size(size)),
            ReadResult::Bytes(bytes) => Ok(IoResult::Bytes(bytes)),
            ReadResult::Suffix { .. } => Err(vortex_err!("Unexpected suffix result in scan")),
        });
        Ok(Some(Completion { owner, request, result }))
    }
}

impl IoSource for Source {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut state = self.state.lock();
        let mut reads = Vec::new();
        let mut pending = Vec::new();
        for request in batch {
            if request.intent != IoIntent::Fetch { continue; }
            let id = ReadId(state.next);
            state.next = state.next.checked_add(1).ok_or_else(|| vortex_err!("Read identity overflow"))?;
            let target = match request.target {
                IoTarget::Size => ReadTarget::Size,
                IoTarget::Range { offset, len, alignment } => ReadTarget::Range { offset, length: len, alignment },
            };
            reads.push(ReadRequest { id, target });
            pending.push((id, (owner, request.request)));
        }
        self.reader()?.submit(&reads)?;
        state.pending.extend(pending);
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        while let Some(completion) = self.reader()?.try_complete()? {
            if let Some(completion) = self.deliver(completion)? { return Ok(Some(completion)); }
        }
        Ok(None)
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        loop {
            let reader = self.reader()?;
            if self.state.lock().pending.is_empty() {
                return Poll::Ready(Err(vortex_err!("Waiting with no fetch outstanding")));
            }
            match reader.poll_complete(cx) {
                Poll::Ready(result) => {
                    if let Some(completion) = self.deliver(result?)? { return Poll::Ready(Ok(completion)); }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        loop {
            if self.state.lock().pending.is_empty() { vortex_bail!("Waiting with no fetch outstanding"); }
            if let Some(completion) = self.deliver(self.reader()?.wait()?)? { return Ok(completion); }
        }
    }

    fn release(&self, owner: IoOwnerId) {
        let mut state = self.state.lock();
        state.pending.retain(|id, (pending_owner, _)| {
            if *pending_owner != owner { return true; }
            if let Ok(reader) = self.reader() { reader.cancel(*id); }
            false
        });
    }

    fn clear(&self) {
        for (id, _) in self.state.lock().pending.drain() {
            if let Ok(reader) = self.reader() { reader.cancel(id); }
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) { self.clear(); }
}
