use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::StreamExt;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, ObjectStoreScheme,
    PutPayload,
};
use vortex_buffer::pool::BufferPool;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult};

use crate::{IoBuf, PerformanceHint, VortexReadAt, VortexWrite};

#[derive(Clone)]
pub struct ObjectStoreReadAt {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
    scheme: Option<ObjectStoreScheme>,
    buffer_pool: Option<BufferPool>,
}

impl ObjectStoreReadAt {
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        location: Path,
        scheme: Option<ObjectStoreScheme>,
    ) -> Self {
        Self {
            object_store,
            location,
            scheme,
            buffer_pool: None,
        }
    }

    pub fn with_buffer_pool(mut self, buffer_pool: BufferPool) -> Self {
        self.buffer_pool = Some(buffer_pool);
        self
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(size = range.end - range.start)))]
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
        let mut buffer = match self.buffer_pool.as_ref() {
            Some(pool) => {
                let mut buffer = pool.get_aligned(alignment);
                buffer.reserve(len);

                buffer
            }
            None => ByteBufferMut::with_capacity_aligned(len, alignment),
        };

        let response = object_store
            .get_opts(
                &location,
                GetOptions {
                    range: Some(GetRange::Bounded(start..end)),
                    ..Default::default()
                },
            )
            .await?;

        let mut buffer = match response.payload {
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

        let reminder = buffer.split_off(len);

        if let Some(pool) = self.buffer_pool.as_ref() {
            pool.put_back(reminder);
        }

        assert!(
            buffer.as_ptr().align_offset(*alignment) == 0,
            "buffer is aligned to {} but requested alignment to {}",
            buffer.alignment(),
            alignment
        );

        Ok(buffer.freeze())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    async fn size(&self) -> io::Result<u64> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        Ok(object_store.head(&location).await?.size as u64)
    }

    fn performance_hint(&self) -> PerformanceHint {
        match &self.scheme {
            Some(ObjectStoreScheme::Local | ObjectStoreScheme::Memory) => PerformanceHint::local(),
            Some(
                ObjectStoreScheme::AmazonS3
                | ObjectStoreScheme::MicrosoftAzure
                | ObjectStoreScheme::GoogleCloudStorage,
            ) => PerformanceHint::object_storage(),
            _ => PerformanceHint::default(),
        }
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
