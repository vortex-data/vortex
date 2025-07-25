// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
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
use vortex_error::VortexResult;

use crate::tokio::{TokioDispatchedIo, TokioReadAt};
use crate::{IoBuf, PerformanceHint, ReadAt, VortexIO, VortexWrite};

#[derive(Clone)]
pub struct ObjectStoreIo {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
    scheme: Option<ObjectStoreScheme>,
}

impl ObjectStoreIo {
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

impl VortexIO for ObjectStoreIo {
    fn performance_hint(&self) -> PerformanceHint {
        match &self.scheme {
            Some(ObjectStoreScheme::Memory) => PerformanceHint::in_memory(),
            Some(ObjectStoreScheme::Local) => PerformanceHint::local(),
            _ => PerformanceHint::object_storage(),
        }
    }

    fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>> {
        // If the file is local, we much prefer to use std::fs::File since object store re-opens
        // the file on every read. This check is a little naive... but we hope that ObjectStore
        // will soon expose the scheme in a way that we can check more thoroughly.
        // See: https://github.com/apache/arrow-rs-object-store/issues/259
        let local_path = std::path::Path::new("/").join(self.location.as_ref());
        if local_path.exists() {
            // TODO(ngates): we could move the open operating into the dispatcher..
            std::fs::File::open(local_path)?.into_read_at()
        } else {
            Ok(Arc::new(TokioDispatchedIo::new(self)))
        }
    }
}

impl TokioReadAt for ObjectStoreIo {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(size = len)))]
    async fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();

        // Instead of calling `ObjectStore::get_range`, we expand the implementation and run it
        // ourselves to avoid a second copy to align the buffer. Instead, we can write directly
        // into the aligned buffer.
        let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);

        let response = object_store
            .get_opts(
                &location,
                GetOptions {
                    range: Some(GetRange::Bounded(offset..offset + len as u64)),
                    ..Default::default()
                },
            )
            .await?;

        let buffer = match response.payload {
            GetResultPayload::File(file, _) => {
                unsafe { buffer.set_len(len) };
                {
                    tokio::task::spawn_blocking(move || {
                        file.read_exact_at(&mut buffer, offset)?;
                        Ok::<_, io::Error>(buffer)
                    })
                    .await
                    .map_err(io::Error::other)??
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

    async fn size(&self) -> VortexResult<u64> {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        let metadata = object_store.head(&location).await?;
        Ok(metadata.size)
    }
}

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

        if self.buffer.len() > CHUNKS_SIZE {
            let mut buffer =
                std::mem::replace(&mut self.buffer, BytesMut::with_capacity(CHUNKS_SIZE)).freeze();
            let mut parts = vec![];

            while buffer.len() > CHUNKS_SIZE {
                let payload = buffer.split_to(CHUNKS_SIZE);
                let part_fut = self
                    .upload
                    .as_mut()
                    .put_part(PutPayload::from_bytes(payload));

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
