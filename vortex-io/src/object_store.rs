// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use bytes::BytesMut;
use futures::future::try_join_all;
use futures_util::StreamExt;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, ObjectStoreScheme,
    PutPayload,
};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult};

use crate::{IoBuf, PerformanceHint, VortexReadAt, VortexWrite};

#[derive(Clone)]
pub struct ObjectStoreReadAt {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
    scheme: Option<ObjectStoreScheme>,
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
        }
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
        let len = usize::try_from(range.end - range.start).vortex_expect("Read can't find usize");

        // Instead of calling `ObjectStore::get_range`, we expand the implementation and run it
        // ourselves to avoid a second copy to align the buffer. Instead, we can write directly
        // into the aligned buffer.
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);

        let response = object_store
            .get_opts(
                &location,
                GetOptions {
                    range: Some(GetRange::Bounded(range.start..range.end)),
                    ..Default::default()
                },
            )
            .await?;

        let buffer = match response.payload {
            GetResultPayload::File(file, _) => {
                // SAFETY: We're setting the length to the exact size we're about to read.
                // The read_exact_at call will either fill the entire buffer or return an error,
                // ensuring no uninitialized memory is exposed.
                unsafe { buffer.set_len(len) };
                #[cfg(feature = "tokio")]
                {
                    tokio::task::spawn_blocking(move || {
                        file.read_exact_at(&mut buffer, range.start)?;
                        Ok::<_, io::Error>(buffer)
                    })
                    .await
                    .map_err(io::Error::other)??
                }
                #[cfg(not(feature = "tokio"))]
                {
                    {
                        file.read_exact_at(&mut buffer, range.start)?;
                        Ok::<_, io::Error>(buffer)
                    }
                    .map_err(io::Error::other)?
                }
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

    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all))]
    async fn size(&self) -> io::Result<u64> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        Ok(object_store.head(&location).await?.size as u64)
    }

    fn performance_hint(&self) -> PerformanceHint {
        match &self.scheme {
            Some(ObjectStoreScheme::Local | ObjectStoreScheme::Memory) => PerformanceHint::local(),
            _ => PerformanceHint::object_storage(),
        }
    }
}

pub struct ObjectStoreWriter {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
}

const CHUNKS_SIZE: usize = 25 * 1024 * 1024;
const MAX_BUFFER_SIZE: usize = 100 * 1024 * 1024; // 100MB max buffer

impl ObjectStoreWriter {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            upload,
            buffer: BytesMut::with_capacity(CHUNKS_SIZE),
        })
    }
}

impl VortexWrite for ObjectStoreWriter {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        // Check if adding this data would exceed max buffer size
        let new_size = self.buffer.len() + buffer.as_slice().len();
        if new_size > MAX_BUFFER_SIZE {
            return Err(io::Error::other(format!(
                "Buffer size would exceed maximum of {} bytes",
                MAX_BUFFER_SIZE
            )));
        }

        self.buffer.extend_from_slice(buffer.as_slice());

        if self.buffer.len() > CHUNKS_SIZE {
            let mut parts = vec![];

            // Split off chunks while buffer is larger than CHUNKS_SIZE
            while self.buffer.len() > CHUNKS_SIZE {
                let chunk = self.buffer.split_to(CHUNKS_SIZE);
                let part_fut = self
                    .upload
                    .as_mut()
                    .put_part(PutPayload::from_bytes(chunk.freeze()));

                parts.push(part_fut);
            }

            try_join_all(parts).await?;
        }

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        let mut buffer = std::mem::take(&mut self.buffer).freeze();
        let mut parts = vec![];

        while !buffer.is_empty() {
            let chunk_size = usize::min(buffer.len(), CHUNKS_SIZE);
            let payload = buffer.split_to(chunk_size);
            let part_fut = self
                .upload
                .as_mut()
                .put_part(PutPayload::from_bytes(payload));

            parts.push(part_fut);
        }

        try_join_all(parts).await?;

        self.upload.complete().await?;
        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::ObjectStore;
    use object_store::path::Path;

    use super::*;

    async fn create_test_store() -> (Arc<object_store::memory::InMemory>, Path) {
        let store = Arc::new(object_store::memory::InMemory::new());
        let location = Path::from("test.bin");
        (store, location)
    }

    // Note: Concurrent writes test removed because &mut self in write_all already ensures
    // exclusive access. Multiple writers would need to be created with separate buffers,
    // which is not the intended use case.

    #[tokio::test]
    async fn test_object_store_writer_max_buffer_size() {
        let (store, location) = create_test_store().await;
        let mut writer = ObjectStoreWriter::new(store, &location).await.unwrap();

        // Try to write more than MAX_BUFFER_SIZE
        let huge_data = vec![0u8; MAX_BUFFER_SIZE + 1];
        let result = writer.write_all(huge_data).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceed maximum"));
    }

    #[tokio::test]
    async fn test_object_store_writer_multiple_flushes() {
        let (store, location) = create_test_store().await;
        let mut writer = ObjectStoreWriter::new(store.clone(), &location)
            .await
            .unwrap();

        // Write and flush multiple times
        for i in 0..3 {
            #[allow(clippy::cast_possible_truncation)]
            let data = vec![i as u8; 100];
            writer.write_all(data).await.unwrap();
            writer.flush().await.unwrap();
        }

        // Verify all data was written
        let result = store.get(&location).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.len(), 300);
    }
}
