// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The session's segment IO service, serving planner-driven scans over files' segment sources.
//!
//! A file is registered by its [`SegmentSource`], usually [`VortexFile::segment_source`], and its
//! size when known, so segment reads go through the same segment cache, request sharing and
//! coalescing read driver as every other read of the file. A size request is answered at submit,
//! and fails for a source registered without one. The intents of
//! segment requests map onto the source's requests:
//!
//! - every segment request calls [`SegmentSource::request`], which for an opened file registers the read
//!   with its coalescing driver without starting it, so even an announcement can widen a read;
//! - a fetch or prefetch is then polled once, after the whole batch is registered, which starts
//!   the read. The session keeps polling it from `poll_ready` until it finishes, so a read whose
//!   first poll had to wait, such as on an asynchronous cache lookup, still starts;
//! - dropping every handle of an announcement drops the source's request, which withdraws the
//!   read.
//!
//! [`VortexFile::segment_source`]: crate::VortexFile::segment_source
//! [`SegmentSource`]: vortex_layout::segments::SegmentSource
//! [`SegmentSource::request`]: vortex_layout::segments::SegmentSource::request

// TODO(joe): give this service a read driver of its own instead of reading through the files'
// v1 segment sources, so it can be optimised (fetches before prefetches, one scheduler across
// files) without changing v1.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::Mutex;
use parking_lot::RwLock;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::segments::FetchTracker;
use vortex_io::segments::IoHandle;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoOwnerId;
use vortex_io::segments::IoRequest;
use vortex_io::segments::RegisteredSource;
use vortex_io::segments::SegmentIoService;
use vortex_io::segments::SegmentIoSession;
use vortex_io::segments::SegmentSourceId;
use vortex_io::segments::SharedResult;
use vortex_io::segments::SourceRequest;
use vortex_io::segments::SourceResult;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Serves the segment reads of every source registered in one session.
#[derive(Clone, Default)]
pub struct FileSegmentIo(Arc<Registry>);

type Source = Arc<dyn SegmentSource>;

/// Source ids are unique across services, so a key submitted to the wrong service's session is
/// rejected as unregistered rather than read from another source.
static NEXT_SOURCE: AtomicU32 = AtomicU32::new(0);

/// A registered source, held weakly so the registration owns it, and its size in bytes if known.
type Registered = (Weak<dyn SegmentSource>, Option<u64>);

#[derive(Default)]
struct Registry {
    sources: RwLock<HashMap<SegmentSourceId, Registered>>,
}

impl Debug for FileSegmentIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSegmentIo")
            .field("sources", &self.0.sources.read().len())
            .finish()
    }
}

impl SessionVar for FileSegmentIo {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl FileSegmentIo {
    /// The service of `session`, created on first use.
    pub fn of(session: &VortexSession) -> Self {
        session.get::<Self>().clone()
    }

    /// Registers `source`, usually a [`VortexFile::segment_source`], whose total length is `size`
    /// bytes if known; size requests of a source registered without one fail. New sessions can read it while any clone of the registration is alive; a
    /// session that has already read it keeps it readable for the session's lifetime.
    ///
    /// [`VortexFile::segment_source`]: crate::VortexFile::segment_source
    pub fn register(&self, source: Source, size: Option<u64>) -> RegisteredSource {
        let id = SegmentSourceId(NEXT_SOURCE.fetch_add(1, Ordering::Relaxed));
        self.0
            .sources
            .write()
            .insert(id, (Arc::downgrade(&source), size));
        let registration = Arc::new(Registration {
            registry: Arc::clone(&self.0),
            id,
            _source: source,
        });
        RegisteredSource::new(Arc::new(self.clone()), id, registration)
    }
}

impl SegmentIoService for FileSegmentIo {
    fn session(&self) -> Arc<dyn SegmentIoSession> {
        Arc::new(FileSession {
            registry: Arc::clone(&self.0),
            sources: Mutex::default(),
            fetches: Mutex::default(),
        })
    }
}

/// Keeps a source registered, and readable, while it lives.
struct Registration {
    registry: Arc<Registry>,
    id: SegmentSourceId,
    _source: Source,
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.registry.sources.write().remove(&self.id);
    }
}

/// One scope's reads through a [`FileSegmentIo`].
struct FileSession {
    registry: Arc<Registry>,
    /// The sources this session has read, so later requests skip the registry, each with the
    /// handle registration that keeps it alive while a handle of it lives.
    sources: Mutex<HashMap<SegmentSourceId, SessionSource>>,
    fetches: Mutex<FetchTracker>,
}

#[derive(Clone)]
struct SessionSource {
    source: Source,
    size: Option<u64>,
    keep_alive: Arc<dyn Any + Send + Sync>,
}

impl FileSession {
    fn source(&self, id: SegmentSourceId) -> VortexResult<SessionSource> {
        let mut sources = self.sources.lock();
        if let Some(source) = sources.get(&id) {
            return Ok(source.clone());
        }
        let (source, size) = self
            .registry
            .sources
            .read()
            .get(&id)
            .and_then(|(source, size)| Some((source.upgrade()?, *size)))
            .ok_or_else(|| vortex_err!("{id:?} is not registered for segment reads"))?;
        let source = SessionSource {
            keep_alive: Arc::new(Arc::clone(&source)),
            source,
            size,
        };
        sources.insert(id, source.clone());
        Ok(source)
    }
}

impl SegmentIoSession for FileSession {
    fn submit(&self, owner: IoOwnerId, batch: &[IoRequest]) -> VortexResult<Vec<IoHandle>> {
        let mut wanted = Vec::new();
        let handles = batch
            .iter()
            .map(|request| {
                let SessionSource {
                    source,
                    size,
                    keep_alive,
                } = self.source(request.key.source)?;
                let segment = match request.key.request {
                    SourceRequest::Size => {
                        let size = size.map(SourceResult::Size).ok_or_else(|| {
                            vortex_err!("the size of {:?} is unknown", request.key.source)
                        });
                        return Ok(IoHandle::ready(size));
                    }
                    SourceRequest::Segment(segment) => segment,
                };
                let result: SharedResult = source
                    .request(SegmentId::from(segment))
                    .map_ok(SourceResult::Bytes)
                    .map_err(Arc::new)
                    .boxed()
                    .shared();
                if request.intent != IoIntent::Announce {
                    wanted.push(result.clone());
                }
                // The handle keeps the source alive while its read may be in flight.
                Ok(IoHandle::new(result, keep_alive))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        // Every read of the batch is registered before any is started, so the driver can choose
        // physical reads with the whole batch in view. The first poll starts a read; later polls
        // through the handles take its result.
        for result in wanted {
            drop(result.now_or_never());
        }
        let mut fetches = self.fetches.lock();
        for (request, handle) in batch.iter().zip(&handles) {
            match request.intent {
                IoIntent::Fetch => fetches.track(owner, handle),
                IoIntent::Prefetch if handle.try_get().is_none() => fetches.track_prefetch(handle),
                IoIntent::Prefetch | IoIntent::Announce => {}
            }
        }
        Ok(handles)
    }

    fn poll_ready(
        &self,
        cx: &mut Context<'_>,
        ready: &mut Vec<IoOwnerId>,
    ) -> Poll<VortexResult<()>> {
        self.fetches.lock().poll_ready(cx, ready)
    }
}

#[cfg(test)]
mod tests;
