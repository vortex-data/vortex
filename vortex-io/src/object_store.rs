use std::future::Future;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, PutPayload,
};
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
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = io::Result<Bytes>> + 'static {
        let object_store = self.object_store.clone();
        let location = self.location.clone();
        Box::pin(async move {
            let read_start: usize = pos.try_into().vortex_expect("pos");
            let read_end: usize = (pos + len).try_into().vortex_expect("pos + len");
            let len: usize = len.try_into().vortex_expect("len does not fit into usize");

            let response = object_store
                .get_opts(
                    &location,
                    GetOptions {
                        range: Some(GetRange::Bounded(read_start..read_end)),
                        ..Default::default()
                    },
                )
                .await?;

            // NOTE: ObjectStore specializes the payload based on if it is backed by a File or if
            //  it's coming from a network stream. Internally they optimize the File implementation
            //  to only perform a single allocation when calling `.bytes().await`, which we
            //  replicate here by emitting the contents directly into our aligned buffer.
            let mut buffer = BytesMut::with_capacity(len);
            match response.payload {
                GetResultPayload::File(file, _) => {
                    unsafe { buffer.set_len(len) };
                    file.read_exact_at(&mut buffer, pos)?;
                }
                GetResultPayload::Stream(mut byte_stream) => {
                    while let Some(bytes) = byte_stream.next().await {
                        buffer.extend_from_slice(&bytes?);
                    }
                }
            }
            Ok(buffer.freeze())
        })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    fn size(&self) -> impl Future<Output = io::Result<u64>> + 'static {
        let object_store = self.object_store.clone();
        let location = self.location.clone();

        Box::pin(async move {
            object_store
                .head(&location)
                .await
                .map(|obj| obj.size as u64)
                .map_err(io::Error::other)
        })
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
