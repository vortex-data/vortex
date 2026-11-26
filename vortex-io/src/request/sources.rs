// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One IO service for every byte source a plan reads.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::FuturesUnordered;
use futures::task::noop_waker_ref;
use parking_lot::Mutex;
use parking_lot::RwLock;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use super::Completion;
use super::IoBatch;
use super::IoIntent;
use super::IoOwnerId;
use super::IoRequest;
use super::IoResult;
use super::IoService;
use super::IoSource;
use super::IoTarget;
use super::OpenTarget;
use super::SourceId;

/// Routes every request of a plan to the service of the source it names.
///
/// A driver takes its sessions from this one service, whatever the number of files its plan
/// reads. Sources join in two ways:
///
/// - Registered ahead of the plan with [`register`](Self::register), which mints their id, such
///   as the file of a single-file scan.
/// - Opened by a planner mid-plan, with an [`IoTarget::Open`] addressed to this service's own
///   [`id`](Self::id). The service opens the [`Openable`](super::Openable), registers the service
///   it opens to, and answers with the new source's id. Opens with one key share one source.
///
/// A request naming an id this service did not mint is an error, never a fallback. Sources stay
/// registered until the service is dropped.
///
/// Cloning shares the registry: every clone mints from and routes to the same sources.
#[derive(Clone)]
pub struct SourcesIoService {
    inner: Arc<Inner>,
}

type SharedOpen = Shared<BoxFuture<'static, Result<SourceId, Arc<VortexError>>>>;

struct Inner {
    /// The id open requests are addressed to.
    id: SourceId,
    next: AtomicU32,
    services: RwLock<HashMap<SourceId, Arc<dyn IoService>>>,
    /// Every open started, by key, shared by later opens of the key.
    opens: Mutex<HashMap<Arc<str>, SharedOpen>>,
}

impl Inner {
    fn mint(&self) -> SourceId {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(id != u32::MAX, "SourcesIoService ran out of source ids");
        SourceId(id)
    }

    fn insert(&self, id: SourceId, io: Arc<dyn IoService>) {
        self.services.write().insert(id, io);
    }

    fn service(&self, id: SourceId) -> VortexResult<Arc<dyn IoService>> {
        self.services
            .read()
            .get(&id)
            .cloned()
            .ok_or_else(|| vortex_err!("{id} is not registered with this plan's sources"))
    }

    /// The open of `target`, started on first use and shared by every later open of its key.
    fn open(self: &Arc<Self>, target: &OpenTarget) -> SharedOpen {
        let key = Arc::clone(target.0.key());
        self.opens
            .lock()
            .entry(key)
            .or_insert_with(|| {
                let inner = Arc::clone(self);
                target
                    .0
                    .open_service()
                    .map_ok(move |io| {
                        let id = inner.mint();
                        inner.insert(id, io);
                        id
                    })
                    .map_err(Arc::new)
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

impl SourcesIoService {
    /// A service with no sources.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                id: SourceId(0),
                next: AtomicU32::new(1),
                services: RwLock::default(),
                opens: Mutex::default(),
            }),
        }
    }

    /// The id requests to open a new source are addressed to. It names no source: reading from
    /// it is an error.
    pub fn id(&self) -> SourceId {
        self.inner.id
    }

    /// Registers a source read through `io`, returning the id its requests name.
    pub fn register(&self, io: Arc<dyn IoService>) -> SourceId {
        let id = self.inner.mint();
        self.inner.insert(id, io);
        id
    }

    /// Registers a source whose service needs its own id to be built, such as a scan whose
    /// planners stamp the id on the requests the service then serves.
    ///
    /// `build` receives the new id and returns what the caller keeps along with the source's
    /// service. When it fails, nothing is registered.
    pub fn register_with<T>(
        &self,
        build: impl FnOnce(SourceId) -> VortexResult<(T, Arc<dyn IoService>)>,
    ) -> VortexResult<T> {
        let id = self.inner.mint();
        let (value, io) = build(id)?;
        self.inner.insert(id, io);
        Ok(value)
    }

    /// The number of sources registered or opened so far.
    pub fn len(&self) -> usize {
        self.inner.services.read().len()
    }

    /// Whether no source is registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SourcesIoService {
    fn default() -> Self {
        Self::new()
    }
}

impl IoService for SourcesIoService {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(SourcesSession {
            inner: Arc::clone(&self.inner),
            state: Mutex::default(),
        })
    }
}

/// One registration scope's view of every source: a session of each source it has read, opened
/// on first use.
struct SourcesSession {
    inner: Arc<Inner>,
    state: Mutex<SessionState>,
}

#[derive(Default)]
struct SessionState {
    sessions: HashMap<SourceId, Arc<dyn IoSource>>,
    /// Fetches submitted to each source's session and not yet completed, by owner.
    in_flight: HashMap<SourceId, HashMap<IoOwnerId, usize>>,
    /// Opens requested by this scope.
    opens: FuturesUnordered<BoxFuture<'static, Completion>>,
    /// Owners that retired with requests in flight, whose completions are dropped. Owner ids
    /// are never reused within a scope.
    released: HashSet<IoOwnerId>,
}

impl SessionState {
    fn session(&mut self, inner: &Inner, id: SourceId) -> VortexResult<Arc<dyn IoSource>> {
        if let Some(session) = self.sessions.get(&id) {
            return Ok(Arc::clone(session));
        }
        let session = inner.service(id)?.session();
        self.sessions.insert(id, Arc::clone(&session));
        Ok(session)
    }

    fn completed(&mut self, id: SourceId, completion: &Completion) {
        let Some(owners) = self.in_flight.get_mut(&id) else {
            return;
        };
        if let Some(count) = owners.get_mut(&completion.owner) {
            *count -= 1;
            if *count == 0 {
                owners.remove(&completion.owner);
            }
        }
        if owners.is_empty() {
            self.in_flight.remove(&id);
        }
    }

    fn is_idle(&self) -> bool {
        self.opens.is_empty() && self.in_flight.is_empty()
    }

    /// The next completion of a live owner, polling opens and every source with fetches in
    /// flight with `cx`.
    fn poll_next(&mut self, cx: &mut Context<'_>) -> VortexResult<Option<Completion>> {
        while let Poll::Ready(Some(completion)) = self.opens.poll_next_unpin(cx) {
            if !self.released.contains(&completion.owner) {
                return Ok(Some(completion));
            }
        }
        let busy: Vec<SourceId> = self.in_flight.keys().copied().collect();
        for id in busy {
            let session = Arc::clone(
                self.sessions
                    .get(&id)
                    .vortex_expect("a source with fetches in flight has a session"),
            );
            loop {
                match session.poll_completion(cx) {
                    Poll::Ready(Ok(completion)) => {
                        self.completed(id, &completion);
                        if self.released.contains(&completion.owner) {
                            if !self.in_flight.contains_key(&id) {
                                break;
                            }
                            continue;
                        }
                        return Ok(Some(completion));
                    }
                    Poll::Ready(Err(error)) => return Err(error),
                    Poll::Pending => break,
                }
            }
        }
        Ok(None)
    }
}

impl IoSource for SourcesSession {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let mut state = self.state.lock();
        let mut by_source: HashMap<SourceId, IoBatch> = HashMap::default();
        for request in batch {
            let IoRequest {
                source,
                intent,
                request: id,
                target,
            } = request;
            if let IoTarget::Open(open) = target {
                if source != self.inner.id {
                    vortex_bail!(
                        "{open:?} is addressed to {source}, not to the sources service {}",
                        self.inner.id
                    );
                }
                let shared = self.inner.open(&open);
                match intent {
                    IoIntent::Fetch => {
                        let key = Arc::clone(open.0.key());
                        state.opens.push(
                            async move {
                                let result = shared
                                    .await
                                    .map(IoResult::Opened)
                                    .map_err(|err| vortex_err!("opening {key}: {err}"));
                                Completion {
                                    owner,
                                    request: id,
                                    result,
                                }
                            }
                            .boxed(),
                        );
                    }
                    // Polling registers the open as wanted, so it starts now.
                    IoIntent::Prefetch => {
                        drop(
                            shared
                                .clone()
                                .poll_unpin(&mut Context::from_waker(noop_waker_ref())),
                        );
                    }
                    IoIntent::Announce => {}
                }
                continue;
            }
            if source == self.inner.id {
                vortex_bail!("{source} names the sources service, which serves only opens");
            }
            by_source.entry(source).or_default().push(IoRequest {
                source,
                intent,
                request: id,
                target,
            });
        }
        for (id, requests) in by_source {
            let session = state.session(&self.inner, id)?;
            let fetches = requests
                .iter()
                .filter(|request| request.intent == IoIntent::Fetch)
                .count();
            session.submit(owner, requests)?;
            if fetches > 0 {
                *state
                    .in_flight
                    .entry(id)
                    .or_default()
                    .entry(owner)
                    .or_default() += fetches;
            }
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        let mut cx = Context::from_waker(noop_waker_ref());
        self.state.lock().poll_next(&mut cx)
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        let mut state = self.state.lock();
        if state.is_idle() {
            return Poll::Ready(Err(vortex_err!(
                "SourcesIoService: waiting with nothing in flight"
            )));
        }
        match state.poll_next(cx) {
            Ok(Some(completion)) => Poll::Ready(Ok(completion)),
            Ok(None) if state.is_idle() => Poll::Ready(Err(vortex_err!(
                "SourcesIoService: waiting with nothing in flight"
            ))),
            Ok(None) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        futures::executor::block_on(futures::future::poll_fn(|cx| self.poll_completion(cx)))
    }

    fn release(&self, owner: IoOwnerId) {
        let mut state = self.state.lock();
        for session in state.sessions.values() {
            session.release(owner);
        }
        let mut had_requests = !state.opens.is_empty();
        state.in_flight.retain(|_, owners| {
            had_requests |= owners.remove(&owner).is_some();
            !owners.is_empty()
        });
        if had_requests {
            state.released.insert(owner);
        }
    }

    fn clear(&self) {
        let mut state = self.state.lock();
        for session in state.sessions.values() {
            session.clear();
        }
        *state = SessionState::default();
    }
}

#[cfg(test)]
mod tests;
