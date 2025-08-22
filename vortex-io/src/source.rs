// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{Handle, Read, VortexRead};
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::FutureExt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult};

pub trait IoSource: 'static + Send + Sync {
    fn open(&self, handle: &Handle) -> Arc<dyn VortexRead>;
}

/// A trait that describes how to process I/O requests for a given source.
/// Vortex runtimes will spawn I/O drivers only when they are needed, partly to free up cores for
/// performing CPU work when there is no I/O, but also to allow the I/O drivers to be non-Send.
pub(crate) trait IoDriver {
    type Data: Send + 'static;

    /// Spawn a local future to process the given a stream of I/O requests until it terminates.
    /// The future should apply reasonable concurrency constraints.
    fn spawn(&self, stream: BoxStream<'static, IoRequest<Self::Data>>) -> impl Future<Output = ()>;
}

/// An I/O request that supports opaque attached data.
pub(crate) struct IoRequest<Data> {
    offset: u64,
    length: usize,
    alignment: Alignment,
    resolve: oneshot::Receiver<VortexResult<ByteBuffer>>,
    data: Data,
}

pub struct FileIo(Arc<File>);

impl FileIo {
    pub fn try_new<P: AsRef<Path>>(path: P) -> VortexResult<Arc<dyn IoSource>> {
        Ok(Arc::new(Self(Arc::new(File::open(path)?))) as _)
    }
}

impl IoSource for FileIo {
    fn open(&self, handle: &Handle) -> Arc<dyn VortexRead> {
        handle.open_file(self.0.clone())
    }
}

pub struct MemoryIo(ByteBuffer);

impl MemoryIo {
    pub fn new(buffer: ByteBuffer) -> Arc<dyn IoSource> {
        Arc::new(Self(buffer))
    }
}

impl IoSource for MemoryIo {
    fn open(&self, _handle: &Handle) -> Arc<dyn VortexRead> {
        Arc::new(self.0.clone())
    }
}

impl VortexRead for ByteBuffer {
    fn read(&self, offset: u64, length: usize, alignment: Alignment) -> Read {
        let offset = usize::try_from(offset).vortex_expect("Offset out of bounds for usize");
        Read::ready(Ok(self
            .slice_unaligned(offset..offset + length)
            .aligned(alignment)))
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.len() as u64;
        async move { Ok(len) }.boxed()
    }
}

#[cfg(feature = "object_store")]
pub struct ObjectStoreIo {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
}

#[cfg(feature = "object_store")]
impl ObjectStoreIo {
    pub fn new(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
    ) -> Arc<dyn IoSource> {
        Arc::new(Self { store, path })
    }
}

#[cfg(feature = "object_store")]
impl IoSource for ObjectStoreIo {
    fn open(&self, runtime: &Handle) -> Arc<dyn VortexRead> {
        todo!()
    }
}
