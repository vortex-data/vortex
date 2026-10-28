// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::SelectAll;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use super::ReadCompletion;
use super::ReadId;
use super::ReadRequest;
use super::ReadResult;
use super::ReadTarget;
use super::VortexReadAt2;
use crate::ReadAtRequest;
use crate::VortexReadAt;

/// Adapts an existing future-based reader, preserving its batched range implementation.
///
/// The caller must drive the runtime required by the wrapped reader. The adapter does not create
/// a runtime or turn asynchronous work into blocking reads.
/// When driving it from a native worker, use a reader with an explicitly captured runtime handle;
/// a handle that looks up a thread-local runtime will not find the caller's runtime there.
pub struct FutureReadAt2 {
    reader: Arc<dyn VortexReadAt>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    queued: VecDeque<ReadRequest>,
    live: HashSet<ReadId>,
    reads: SelectAll<BoxStream<'static, ReadCompletion>>,
    active: usize,
    waker: Option<Waker>,
}

impl FutureReadAt2 {
    /// Creates an independent completion scope over `reader`.
    pub fn new(reader: Arc<dyn VortexReadAt>) -> Self {
        Self {
            reader,
            state: Mutex::default(),
        }
    }

    fn poll(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>> {
        let mut state = self.state.lock();
        state.waker = Some(cx.waker().clone());
        loop {
            let available = self
                .reader
                .concurrency()
                .max(1)
                .saturating_sub(state.active);
            let count = available.min(state.queued.len());
            if count != 0 {
                let batch: Vec<_> = state.queued.drain(..count).collect();
                state.active += count;
                let mut ranges = Vec::new();
                for request in batch {
                    match request.target {
                        ReadTarget::Range {
                            offset,
                            length,
                            alignment,
                        } => {
                            ranges
                                .push((request.id, ReadAtRequest::new(offset, length, alignment)));
                        }
                        ReadTarget::Size => {
                            let read = self.reader.size();
                            state.reads.push(
                                stream::once(async move {
                                    ReadCompletion {
                                        id: request.id,
                                        result: read.await.map(ReadResult::Size),
                                    }
                                })
                                .boxed(),
                            );
                        }
                        ReadTarget::Suffix { length, alignment } => {
                            let reader = Arc::clone(&self.reader);
                            state.reads.push(
                                stream::once(async move {
                                    ReadCompletion {
                                        id: request.id,
                                        result: reader.read_suffix(length, alignment).await.map(
                                            |(size, bytes)| ReadResult::Suffix { size, bytes },
                                        ),
                                    }
                                })
                                .boxed(),
                            );
                        }
                    }
                }
                if !ranges.is_empty() {
                    let requests = ranges
                        .iter()
                        .map(|(_, request)| *request)
                        .collect::<Vec<_>>();
                    let mut results = self.reader.read_ranges(requests.into());
                    state.reads.push(async_stream::stream! {
                        let mut remaining: Vec<_> = ranges.into_iter().map(Some).collect();
                        while let Some((request, result)) = results.next().await {
                            let Some(slot) = remaining.iter_mut().find(|entry| {
                                entry.as_ref().is_some_and(|(_, pending)| *pending == request)
                            }) else {
                                continue;
                            };
                            if let Some((id, expected)) = slot.take() {
                                let result = result.and_then(|bytes| {
                                    if bytes.len() != expected.length {
                                        vortex_bail!("read_ranges returned {} bytes, expected {}", bytes.len(), expected.length);
                                    }
                                    Ok(ReadResult::Bytes(bytes))
                                });
                                yield ReadCompletion { id, result };
                            }
                        }
                        for (id, _) in remaining.into_iter().flatten() {
                            let error = vortex_err!("read_ranges ended before request {id:?} completed");
                            yield ReadCompletion { id, result: Err(error) };
                        }
                    }.boxed());
                }
            }
            match state.reads.poll_next_unpin(cx) {
                Poll::Ready(Some(completion)) => {
                    state.active -= 1;
                    if state.live.remove(&completion.id) {
                        return Poll::Ready(Ok(completion));
                    }
                }
                Poll::Ready(None) | Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl VortexReadAt2 for FutureReadAt2 {
    fn submit(&self, requests: &[ReadRequest]) -> VortexResult<()> {
        let mut state = self.state.lock();
        let mut ids: HashSet<ReadId> = HashSet::default();
        for request in requests {
            if state.live.contains(&request.id) || !ids.insert(request.id) {
                vortex_bail!("Duplicate read identity {:?}", request.id);
            }
            if let ReadTarget::Range { offset, length, .. } = request.target
                && offset.checked_add(length as u64).is_none()
            {
                vortex_bail!("Read range overflows u64");
            }
        }
        state.live.extend(ids);
        state.queued.extend(requests.iter().copied());
        let waker = state.waker.take();
        drop(state);
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    fn try_complete(&self) -> VortexResult<Option<ReadCompletion>> {
        match self.poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(result) => result.map(Some),
            Poll::Pending => Ok(None),
        }
    }

    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>> {
        self.poll(cx)
    }

    fn cancel(&self, id: ReadId) {
        let mut state = self.state.lock();
        state.live.remove(&id);
        state.queued.retain(|request| request.id != id);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> VortexResult<ReadCompletion> {
        futures::executor::block_on(std::future::poll_fn(|cx| self.poll_complete(cx)))
    }
}
