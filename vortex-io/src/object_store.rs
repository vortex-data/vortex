use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::StreamExt;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, PutPayload,
};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
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
    async fn read_byte_range(
        &self,
        range: Range<u64>,
        alignment: Alignment,
    ) -> io::Result<ByteBuffer> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        let start = usize::try_from(range.start).vortex_expect("range.start");
        let end = usize::try_from(range.end).vortex_expect("range.end");
        let len: usize = end - start;

        // Instead of calling `ObjectStore::get_range`, we expand the implementation and run it
        // ourselves to avoid a second copy to align the buffer. Instead, we can write directly
        // into the aligned buffer.
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);

        let response = object_store
            .get_opts(
                &location,
                GetOptions {
                    range: Some(GetRange::Bounded(start..end)),
                    ..Default::default()
                },
            )
            .await?;

        let buffer = match response.payload {
            GetResultPayload::File(file, _) => {
                unsafe { buffer.set_len(len) };
                tokio::task::spawn_blocking(move || {
                    file.read_exact_at(&mut buffer, range.start)?;
                    Ok::<_, io::Error>(buffer)
                })
                .await
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))??
            }
            GetResultPayload::Stream(mut byte_stream) => {
                while let Some(bytes) = byte_stream.next().await {
                    buffer.extend_from_slice(&bytes?);
                }
                buffer
            }
        };

        Ok(buffer.freeze())
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
