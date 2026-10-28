// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use super::Completion;
use super::IoBatch;
use super::IoIntent;
use super::IoOwnerId;
use super::IoRequestId;
use super::IoResult;
use super::IoService;
use super::IoSource;
use super::IoTarget;

/// Retains completed reads within each registration scope, up to a requested-byte budget.
///
/// Consumers in the same scope can reuse bytes after an earlier consumer retires. Each scope
/// owns a separate cache; clearing or dropping it releases its retained buffers. The first
/// reads to finish occupy the budget. Larger reads and reads exceeding the budget pass through.
/// Buffer slices may retain a larger underlying allocation than their requested length.
pub struct ScopedIoCache {
    inner: Arc<dyn IoService>,
    budget: usize,
}

impl ScopedIoCache {
    /// Wraps an immutable byte source, with a separate `budget` for every session.
    pub fn new(inner: Arc<dyn IoService>, budget: usize) -> Self {
        Self { inner, budget }
    }
}

impl IoService for ScopedIoCache {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(Source {
            inner: self.inner.session(),
            budget: self.budget,
            state: Mutex::default(),
        })
    }
}

struct Source {
    inner: Arc<dyn IoSource>,
    budget: usize,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    bytes: usize,
    cached: HashMap<IoTarget, IoResult>,
    pending: HashMap<(IoOwnerId, IoRequestId), IoTarget>,
    ready: VecDeque<Completion>,
    waker: Option<Waker>,
}

impl Source {
    fn cached(&self) -> Option<Completion> {
        let mut state = self.state.lock();
        let result = state.ready.pop_front()?;
        state.pending.remove(&(result.owner, result.request));
        Some(result)
    }

    fn complete(&self, completion: Completion) -> Option<Completion> {
        let mut state = self.state.lock();
        let target = state.pending.remove(&(completion.owner, completion.request))?;
        let len = match target { IoTarget::Size => 0, IoTarget::Range { len, .. } => len };
        if let Ok(result) = &completion.result
            && result.matches(&target)
            && len <= self.budget.saturating_sub(state.bytes)
            && !state.cached.contains_key(&target)
        {
            state.cached.insert(target, result.clone());
            state.bytes += len;
        }
        Some(completion)
    }
}

impl IoSource for Source {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut state = self.state.lock();
        let mut reads = Vec::with_capacity(batch.len());
        for request in batch {
            if request.intent == IoIntent::Fetch {
                state.pending.insert((owner, request.request), request.target);
                if let Some(result) = state.cached.get(&request.target).cloned() {
                    state.ready.push_back(Completion { owner, request: request.request, result: Ok(result) });
                    continue;
                }
            }
            reads.push(request);
        }
        self.inner.submit(owner, reads)?;
        let waker = if state.ready.is_empty() { None } else { state.waker.take() };
        drop(state);
        if let Some(waker) = waker { waker.wake(); }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        if let Some(completion) = self.cached() { return Ok(Some(completion)); }
        while let Some(completion) = self.inner.poll()? {
            if let Some(completion) = self.complete(completion) { return Ok(Some(completion)); }
        }
        Ok(None)
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        self.state.lock().waker = Some(cx.waker().clone());
        if let Some(completion) = self.cached() { return Poll::Ready(Ok(completion)); }
        loop {
            match self.inner.poll_completion(cx) {
                Poll::Ready(result) => {
                    if let Some(completion) = self.complete(result?) { return Poll::Ready(Ok(completion)); }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        if let Some(completion) = self.cached() { return Ok(completion); }
        loop {
            if let Some(completion) = self.complete(self.inner.wait()?) { return Ok(completion); }
        }
    }

    fn release(&self, owner: IoOwnerId) {
        let mut state = self.state.lock();
        state.pending.retain(|(id, _), _| *id != owner);
        state.ready.retain(|completion| completion.owner != owner);
        self.inner.release(owner);
    }

    fn clear(&self) {
        *self.state.lock() = State::default();
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests;
