// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::executor::block_on;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::IoBuf;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::VortexWrite;
use crate::compat::Compat;

/// A `VortexReadAt` whose synchronous future/stream constructors require a Tokio context.
struct SyncTokioReadAt;

impl VortexReadAt for SyncTokioReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let _handle = tokio::runtime::Handle::current();
        async { Ok(4) }.boxed()
    }

    fn read_at(
        &self,
        _offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let _handle = tokio::runtime::Handle::current();
        async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![0u8; length]))) }.boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        let _handle = tokio::runtime::Handle::current();
        let results: Vec<_> = requests
            .iter()
            .copied()
            .map(|req| {
                (
                    req,
                    Ok(BufferHandle::new_host(ByteBuffer::from(vec![
                        0u8;
                        req.length
                    ]))),
                )
            })
            .collect();
        futures::stream::iter(results).boxed()
    }
}

struct SyncTokioWrite;

impl VortexWrite for SyncTokioWrite {
    fn write_all<B: IoBuf>(
        &mut self,
        buffer: B,
    ) -> impl Future<Output = std::io::Result<B>> + Send {
        let _handle = tokio::runtime::Handle::current();
        async move { Ok(buffer) }
    }

    fn flush(&mut self) -> impl Future<Output = std::io::Result<()>> + Send {
        let _handle = tokio::runtime::Handle::current();
        async { Ok(()) }
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> + Send {
        let _handle = tokio::runtime::Handle::current();
        async { Ok(()) }
    }
}

#[test]
fn test_compat_read_at_enters_tokio_on_construction() -> VortexResult<()> {
    assert!(tokio::runtime::Handle::try_current().is_err());
    let compat = Compat::new(SyncTokioReadAt);
    assert_eq!(block_on(compat.size())?, 4);
    assert_eq!(block_on(compat.read_at(0, 4, Alignment::none()))?.len(), 4);
    let reqs: Arc<[ReadAtRequest]> = Arc::from([ReadAtRequest::new(0, 4, Alignment::none())]);
    let results: Vec<_> = block_on(compat.read_ranges(reqs).collect());
    assert_eq!(results.len(), 1);
    for (_req, buf) in results {
        assert_eq!(buf?.len(), 4);
    }
    Ok(())
}

#[test]
fn test_compat_write_enters_tokio_on_construction() -> std::io::Result<()> {
    assert!(tokio::runtime::Handle::try_current().is_err());
    let mut compat = Compat::new(SyncTokioWrite);
    let buf = block_on(compat.write_all(vec![1u8, 2, 3]))?;
    assert_eq!(buf, vec![1u8, 2, 3]);
    block_on(compat.flush())?;
    block_on(compat.shutdown())?;
    Ok(())
}
