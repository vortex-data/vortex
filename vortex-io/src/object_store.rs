// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use bytes::BytesMut;
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt};
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
    #[tracing::instrument(skip_all, fields(size = range.end - range.start))]
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

    #[tracing::instrument(skip_all)]
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

/// Adapter type to write data through a [`ObjectStore`] instace.
///
/// After writing, the caller must make sure to call `shutdonw`, in order to ensure the data is actually persisted.
pub struct ObjectStoreWriter {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
}

const CHUNKS_SIZE: usize = 25 * 1024 * 1024;

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
        self.buffer.extend_from_slice(buffer.as_slice());
        let parts = FuturesUnordered::new();

        // Split off chunks while buffer is larger than CHUNKS_SIZE
        while self.buffer.len() > CHUNKS_SIZE {
            let payload = self.buffer.split_to(CHUNKS_SIZE).freeze();
            let part_fut = self.upload.put_part(PutPayload::from_bytes(payload));

            parts.push(part_fut);
        }

        parts.try_collect::<Vec<_>>().await?;

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        let parts = FuturesUnordered::new();

        while self.buffer.len() > CHUNKS_SIZE {
            let payload = self.buffer.split_to(CHUNKS_SIZE).freeze();
            let part_fut = self.upload.put_part(PutPayload::from_bytes(payload));

            parts.push(part_fut);
        }

        parts.try_collect::<Vec<_>>().await?;

        Ok(())
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        self.flush().await?;

        if !self.buffer.is_empty() {
            let payload = std::mem::take(&mut self.buffer).freeze();
            self.upload
                .put_part(PutPayload::from_bytes(payload))
                .await?;
        }

        self.upload.complete().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::ObjectStore;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;

    // Note: Concurrent writes test removed because &mut self in write_all already ensures
    // exclusive access. Multiple writers would need to be created with separate buffers,
    // which is not the intended use case.

    #[tokio::test]
    #[rstest]
    #[case(100)]
    #[case(8 * 1024 * 1024)]
    #[case(25 * 1024 * 1024)]
    #[case(26 * 1024 * 1024)]
    async fn test_object_store_writer_multiple_flushes(
        #[case] chunk_size: usize,
    ) -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let local_store =
            Arc::new(LocalFileSystem::new_with_prefix(temp_dir.path())?) as Arc<dyn ObjectStore>;
        let memory_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let location = Path::from("test.bin");

        for test_store in [memory_store, local_store] {
            let mut writer = ObjectStoreWriter::new(test_store.clone(), &location).await?;

            #[expect(clippy::cast_possible_truncation)]
            let data = (0..3)
                .map(|i| vec![i as u8; chunk_size])
                .collect::<Vec<_>>();

            // Write and flush multiple times
            for i in 0..3 {
                let data = data[i].clone();
                writer.write_all(data).await?;
                writer.flush().await?;
            }

            // Shutdown the writer to make sure data actually gets persisted.
            writer.shutdown().await?;

            // Verify all data was written
            let result = test_store.get(&location).await?;
            let bytes = result.bytes().await?;

            let expected_data = itertools::concat(data.into_iter());
            assert_eq!(bytes, expected_data);
        }

        Ok(())
    }
}
