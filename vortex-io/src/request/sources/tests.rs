// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::*;
use crate::VortexReadAt;
use crate::request::IoRequestId;
use crate::request::Openable;
use crate::request::ReadAtIoService;
use crate::request::ReadAtOpenable;

fn bytes(fill: u8) -> Arc<dyn VortexReadAt> {
    Arc::new(ByteBuffer::from(vec![fill; 16]))
}

fn fetch(source: SourceId, id: u32, target: IoTarget) -> IoRequest {
    IoRequest {
        source,
        intent: IoIntent::Fetch,
        request: IoRequestId(id),
        target,
    }
}

fn open(sources: &SourcesIoService, id: u32, openable: impl Openable) -> IoRequest {
    fetch(sources.id(), id, IoTarget::Open(OpenTarget::new(openable)))
}

fn host(result: VortexResult<IoResult>) -> VortexResult<Vec<u8>> {
    match result? {
        IoResult::Bytes(bytes) => Ok(bytes.try_to_host_sync()?.to_vec()),
        other => Err(vortex_err!("expected bytes, got {}", other.kind())),
    }
}

#[test]
fn routes_each_request_to_its_source() -> VortexResult<()> {
    let sources = SourcesIoService::new();
    let a = sources.register(Arc::new(ReadAtIoService::new(bytes(1))));
    let b = sources.register(Arc::new(ReadAtIoService::new(bytes(2))));
    assert_ne!(a, b);
    assert_ne!(a, sources.id());
    let session = sources.session();
    session.submit(
        IoOwnerId(7),
        vec![
            fetch(a, 0, IoTarget::range(0, 2)),
            fetch(b, 1, IoTarget::range(4, 3)),
        ],
    )?;
    let mut got = vec![];
    for _ in 0..2 {
        let completion = session.wait()?;
        assert_eq!(completion.owner, IoOwnerId(7));
        got.push((completion.request, host(completion.result)?));
    }
    got.sort_by_key(|(id, _)| id.0);
    assert_eq!(
        got,
        vec![(IoRequestId(0), vec![1, 1]), (IoRequestId(1), vec![2, 2, 2])]
    );
    assert!(session.poll()?.is_none());
    assert!(session.wait().is_err(), "nothing is in flight");
    Ok(())
}

#[test]
fn rejects_ids_it_did_not_mint() -> VortexResult<()> {
    let other = SourcesIoService::new();
    let foreign = other.register(Arc::new(ReadAtIoService::new(bytes(0))));
    let sources = SourcesIoService::new();
    let session = sources.session();
    // Both services mint 1 first, so the foreign id is only caught because nothing is registered.
    assert!(
        session
            .submit(IoOwnerId(0), vec![fetch(foreign, 0, IoTarget::Size)])
            .is_err()
    );
    assert!(
        session
            .submit(IoOwnerId(0), vec![fetch(sources.id(), 0, IoTarget::Size)])
            .is_err(),
        "the service's own id serves only opens"
    );
    Ok(())
}

/// Counts how often it is opened.
#[derive(Clone)]
struct CountingOpenable {
    inner: ReadAtOpenable,
    opened: Arc<AtomicUsize>,
}

impl fmt::Debug for CountingOpenable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl Openable for CountingOpenable {
    fn key(&self) -> &Arc<str> {
        self.inner.key()
    }

    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
        self.opened.fetch_add(1, Ordering::Relaxed);
        self.inner.open()
    }
}

#[test]
fn opens_a_source_mid_plan_and_reads_it() -> VortexResult<()> {
    let sources = SourcesIoService::new();
    let opened = Arc::new(AtomicUsize::new(0));
    let openable = CountingOpenable {
        inner: ReadAtOpenable::new("mem://a", bytes(9)),
        opened: Arc::clone(&opened),
    };
    let session = sources.session();
    session.submit(
        IoOwnerId(1),
        vec![
            open(&sources, 0, openable.clone()),
            open(&sources, 1, openable),
        ],
    )?;
    let mut ids = vec![];
    for _ in 0..2 {
        match session.wait()?.result? {
            IoResult::Opened(id) => ids.push(id),
            other => panic!("expected an open, got {}", other.kind()),
        }
    }
    assert_eq!(ids[0], ids[1], "one key opens one source");
    assert_eq!(opened.load(Ordering::Relaxed), 1);
    assert_eq!(sources.len(), 1);

    // Another session of the same service reads the opened source.
    let other = sources.session();
    other.submit(IoOwnerId(2), vec![fetch(ids[0], 5, IoTarget::range(0, 1))])?;
    assert_eq!(host(other.wait()?.result)?, vec![9]);
    Ok(())
}

#[test]
fn a_failed_open_fails_the_request() -> VortexResult<()> {
    #[derive(Debug)]
    struct Missing(Arc<str>);
    impl Openable for Missing {
        fn key(&self) -> &Arc<str> {
            &self.0
        }
        fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
            async { Err(vortex_err!("no such file")) }.boxed()
        }
    }
    let sources = SourcesIoService::new();
    let session = sources.session();
    session.submit(IoOwnerId(0), vec![open(&sources, 0, Missing("gone".into()))])?;
    let error = session.wait()?.result.err().map(|err| err.to_string());
    assert!(error.is_some_and(|err| err.contains("gone") && err.contains("no such file")));
    assert!(sources.is_empty());
    Ok(())
}

#[test]
fn released_owners_get_no_completions() -> VortexResult<()> {
    let sources = SourcesIoService::new();
    let a = sources.register(Arc::new(ReadAtIoService::new(bytes(1))));
    let session = sources.session();
    session.submit(IoOwnerId(1), vec![fetch(a, 0, IoTarget::Size)])?;
    session.submit(IoOwnerId(2), vec![fetch(a, 0, IoTarget::Size)])?;
    session.release(IoOwnerId(1));
    let completion = session.wait()?;
    assert_eq!(completion.owner, IoOwnerId(2));
    assert!(session.poll()?.is_none());
    Ok(())
}

#[test]
fn register_with_hands_the_id_to_the_builder() -> VortexResult<()> {
    let sources = SourcesIoService::new();
    let id = sources.register_with(|id| {
        Ok((id, Arc::new(ReadAtIoService::new(bytes(3))) as Arc<dyn IoService>))
    })?;
    let session = sources.session();
    session.submit(IoOwnerId(0), vec![fetch(id, 0, IoTarget::Size)])?;
    assert!(matches!(session.wait()?.result?, IoResult::Size(16)));
    assert!(
        sources
            .register_with(|_| -> VortexResult<((), Arc<dyn IoService>)> {
                Err(vortex_err!("no"))
            })
            .is_err()
    );
    assert_eq!(sources.len(), 1);
    Ok(())
}
