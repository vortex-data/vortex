// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use bytes::BytesMut;
use futures::TryStreamExt;
use futures::stream::FuturesUnordered;
use object_store::MultipartUpload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::path::Path;
use vortex_error::VortexResult;

use crate::IoBuf;
use crate::VortexWrite;

/// Adapter type to write data through a [`ObjectStore`] instance.
///
/// After writing, the caller must make sure to call `shutdown`, in order to ensure the data is actually persisted.
pub struct ObjectStoreWrite {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
    put_result: Option<PutResult>,
}

const CHUNK_SIZE: usize = 16 * 1024 * 1024;
const BUFFER_SIZE: usize = 128 * 1024 * 1024;

impl ObjectStoreWrite {
    pub async fn new(object_store: Arc<dyn ObjectStore>, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            upload,
            buffer: BytesMut::with_capacity(CHUNK_SIZE),
            put_result: None,
        })
    }

    pub fn put_result(&self) -> Option<&PutResult> {
        self.put_result.as_ref()
    }
}

impl VortexWrite for ObjectStoreWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.buffer.extend_from_slice(buffer.as_slice());
        let parts = FuturesUnordered::new();

        // If the buffer is full
        if self.buffer.len() > BUFFER_SIZE {
            // Split off chunks while buffer is larger than CHUNKS_SIZE
            while self.buffer.len() > CHUNK_SIZE {
                let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
                let part_fut = self.upload.put_part(PutPayload::from_bytes(payload));

                parts.push(part_fut);
            }
        }

        parts.try_collect::<Vec<_>>().await?;

        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        let parts = FuturesUnordered::new();

        while self.buffer.len() > CHUNK_SIZE {
            let payload = self.buffer.split_to(CHUNK_SIZE).freeze();
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

        self.put_result = Some(self.upload.complete().await?);
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
            let mut writer = ObjectStoreWrite::new(test_store.clone(), &location).await?;

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
