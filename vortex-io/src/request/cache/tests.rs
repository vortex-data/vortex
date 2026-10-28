// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use super::ScopedIoCache;
use crate::VortexReadAt;
use crate::read_at2::FutureReadAt2;
use crate::read_at2::ReadAt2IoService;
use crate::request::IoIntent;
use crate::request::IoOwnerId;
use crate::request::IoRequest;
use crate::request::IoRequestId;
use crate::request::IoResult;
use crate::request::IoService;
use crate::request::IoSource;
use crate::request::IoTarget;

struct Counting(AtomicUsize);

impl VortexReadAt for Counting {
    fn concurrency(&self) -> usize {
        1
    }
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        ByteBuffer::from(vec![1, 2, 3, 4]).size()
    }
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        ByteBuffer::from(vec![1, 2, 3, 4]).read_at(offset, length, alignment)
    }
}

fn fetch(source: &dyn IoSource, owner: u64, offset: u64, length: usize) -> VortexResult<IoResult> {
    source.submit(
        IoOwnerId(owner),
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::range(offset, length),
        }],
    )?;
    let result = source.wait()?;
    assert_eq!(result.owner, IoOwnerId(owner));
    result.result
}

#[test]
fn reads_survive_consumer_retirement_but_not_scope_cleanup() -> VortexResult<()> {
    let read = Arc::new(Counting(AtomicUsize::new(0)));
    let factory_read = Arc::clone(&read);
    let cache = ScopedIoCache::new(
        Arc::new(ReadAt2IoService::new(Arc::new(move || {
            Ok(Arc::new(FutureReadAt2::new(Arc::clone(&factory_read) as _)))
        }))),
        4,
    );
    let source = cache.session();
    fetch(source.as_ref(), 1, 0, 4)?;
    source.release(IoOwnerId(1));
    let IoResult::Bytes(bytes) = fetch(source.as_ref(), 2, 0, 4)? else {
        panic!("Expected bytes");
    };
    assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[1, 2, 3, 4]);
    assert_eq!(read.0.load(Ordering::Relaxed), 1);
    source.clear();
    fetch(source.as_ref(), 3, 0, 4)?;
    assert_eq!(read.0.load(Ordering::Relaxed), 2);
    fetch(cache.session().as_ref(), 4, 0, 4)?;
    assert_eq!(read.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[test]
fn budget_errors_and_cancelled_hits_preserve_delivery() -> VortexResult<()> {
    let read = Arc::new(Counting(AtomicUsize::new(0)));
    let factory_read = Arc::clone(&read);
    let cache = ScopedIoCache::new(
        Arc::new(ReadAt2IoService::new(Arc::new(move || {
            Ok(Arc::new(FutureReadAt2::new(Arc::clone(&factory_read) as _)))
        }))),
        2,
    );
    let source = cache.session();
    fetch(source.as_ref(), 1, 0, 2)?;
    for owner in 2..4 {
        fetch(source.as_ref(), owner, 2, 2)?;
    }
    assert_eq!(read.0.load(Ordering::Relaxed), 3);
    for owner in 4..6 {
        assert!(fetch(source.as_ref(), owner, 10, 2).is_err());
    }
    assert_eq!(read.0.load(Ordering::Relaxed), 5);
    source.submit(
        IoOwnerId(6),
        vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::range(0, 2),
        }],
    )?;
    source.release(IoOwnerId(6));
    assert!(source.poll()?.is_none());
    fetch(source.as_ref(), 7, 0, 2)?;
    assert_eq!(read.0.load(Ordering::Relaxed), 5);
    Ok(())
}
