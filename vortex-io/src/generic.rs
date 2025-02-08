use std::future::Future;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use futures_util::TryFutureExt;
use object_store::path::Path;
use object_store::ObjectStore;
use tokio::task::spawn_blocking;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::TokioFile;

/// A generic trait for readable objects.
pub trait GenericRead: 'static + Clone {
    fn read_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> impl Future<Output = VortexResult<ByteBuffer>>;

    fn size(&self) -> impl Future<Output = VortexResult<u64>>;
}

impl GenericRead for ByteBuffer {
    async fn read_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let start = usize::try_from(range.start).vortex_expect("start does not fit into usize");
        let end = usize::try_from(range.end).vortex_expect("start does not fit into usize");
        Ok(self.slice(start..end).aligned(alignment))
    }

    async fn size(&self) -> VortexResult<u64> {
        Ok(self.len() as u64)
    }
}

impl GenericRead for TokioFile {
    async fn read_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let len = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("range does not fit into usize"))?;
        let this = self.clone();

        spawn_blocking(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
            unsafe { buffer.set_len(len) };
            this.read_exact_at(buffer.as_mut_slice(), range.start)?;
            Ok(buffer.freeze())
        })
        .await
        .map_err(|e| vortex_err!("TokioFile error {}", e))?
    }

    async fn size(&self) -> VortexResult<u64> {
        Ok(self.metadata()?.len())
    }
}

#[derive(Clone)]
pub struct ObjectStoreRead {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
}

impl ObjectStoreRead {
    pub fn new(object_store: Arc<dyn ObjectStore>, location: Path) -> Self {
        Self {
            object_store,
            location,
        }
    }
}

impl GenericRead for ObjectStoreRead {
    async fn read_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let start = usize::try_from(range.start).vortex_expect("start does not fit into usize");
        let end = usize::try_from(range.end).vortex_expect("end does not fit into usize");

        self.object_store
            .get_range(&self.location, start..end)
            .map_ok(|bytes| ByteBuffer::from(bytes).aligned(alignment))
            .map_err(Into::into)
            .await
    }

    async fn size(&self) -> VortexResult<u64> {
        self.object_store
            .head(&self.location)
            .map_ok(|metadata| metadata.size as u64)
            .map_err(Into::into)
            .await
    }
}
