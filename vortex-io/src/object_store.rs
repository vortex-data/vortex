use std::io;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use vortex_error::{VortexExpect, VortexResult};

use crate::{IoBuf, VortexReadAt, VortexWrite};

#[derive(Clone)]
pub struct ObjectStoreReadAt {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
}

impl ObjectStoreReadAt {
    pub fn new(object_store: Arc<dyn ObjectStore>, location: Path) -> Self {
        Self {
            object_store,
            location,
        }
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    async fn read_byte_range(&self, range: Range<u64>) -> io::Result<Bytes> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        let start = usize::try_from(range.start).vortex_expect("range.start");
        let end = usize::try_from(range.end).vortex_expect("range.end");
        object_store
            .get_range(&location, start..end)
            .await
            .map_err(Into::into)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    async fn size(&self) -> io::Result<u64> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        Ok(object_store.head(&location).await?.size as u64)
    }
}

pub struct ObjectStoreWriter {
    upload: Box<dyn MultipartUpload>,
}

impl ObjectStoreWriter {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(&location).await?;
        Ok(Self { upload })
    }
}

impl VortexWrite for ObjectStoreWriter {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        const CHUNKS_SIZE: usize = 25 * 1024 * 1024;

        for chunk in buffer.as_slice().chunks(CHUNKS_SIZE) {
            let payload = Bytes::copy_from_slice(chunk);
            self.upload
                .as_mut()
                .put_part(PutPayload::from_bytes(payload))
                .await?;
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.upload.complete().await?;
        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        Ok(())
    }
}
