// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::io::TokioWrite;
use bytes::BytesMut;
use futures::future::try_join_all;
use object_store::path::Path;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

const CHUNKS_SIZE: usize = 25 * 1024 * 1024;

pub(crate) struct ObjectStoreWrite {
    upload: Box<dyn MultipartUpload>,
    buffer: BytesMut,
}

impl ObjectStoreWrite {
    pub(crate) async fn new(object_store: &dyn ObjectStore, location: &Path) -> VortexResult<Self> {
        let upload = object_store.put_multipart(location).await?;
        Ok(Self {
            upload,
            buffer: BytesMut::with_capacity(CHUNKS_SIZE),
        })
    }
}

impl TokioWrite for ObjectStoreWrite {
    async fn write(&mut self, buffer: ByteBuffer) -> VortexResult<ByteBuffer> {
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

    async fn flush(&mut self) -> VortexResult<()> {
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

        // TODO(ngates): shouldn't this happen in shutdown, rather than flush?
        self.upload.complete().await?;
        Ok(())
    }

    async fn shutdown(&mut self) -> VortexResult<()> {
        Ok(())
    }
}
