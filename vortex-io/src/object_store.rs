use std::future::Future;
use std::ops::Range;
use std::sync::Arc;
use std::{io, mem};

use bytes::Bytes;
use futures_util::StreamExt;
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, WriteMultipart};
use vortex_buffer::io_buf::IoBuf;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::aligned::AlignedBytesMut;
use crate::{VortexBufReader, VortexReadAt, VortexWrite, ALIGNMENT};

pub trait ObjectStoreExt {
    fn vortex_read(
        &self,
        location: &Path,
        range: Range<usize>,
    ) -> impl Future<Output = VortexResult<VortexBufReader<impl VortexReadAt>>>;

    fn vortex_reader(&self, location: &Path) -> impl VortexReadAt;

    fn vortex_writer(
        &self,
        location: &Path,
    ) -> impl Future<Output = VortexResult<impl VortexWrite>>;
}

impl ObjectStoreExt for Arc<dyn ObjectStore> {
    async fn vortex_read(
        &self,
        location: &Path,
        range: Range<usize>,
    ) -> VortexResult<VortexBufReader<impl VortexReadAt>> {
        let bytes = self.get_range(location, range).await?;
        Ok(VortexBufReader::new(Buffer::from(bytes)))
    }

    fn vortex_reader(&self, location: &Path) -> impl VortexReadAt {
        ObjectStoreReadAt::new(self.clone(), location.clone())
    }

    async fn vortex_writer(&self, location: &Path) -> VortexResult<impl VortexWrite> {
        Ok(ObjectStoreWriter::new(WriteMultipart::new_with_chunk_size(
            self.put_multipart(location).await?,
            10 * 1024 * 1024,
        )))
    }
}

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
            let start_range = pos as usize;

            let mut buf = AlignedBytesMut::<ALIGNMENT>::with_capacity(len as _);

            let get_range = start_range..(start_range + len as usize);
            let response = object_store
                .get_opts(
                    &location,
                    GetOptions {
                        range: Some(GetRange::Bounded(get_range)),
                        ..Default::default()
                    },
                )
                .await?;

            let mut byte_stream = response.into_stream();
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                buf.extend_from_slice(&bytes);
            }

            Ok(buf.freeze())
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
    multipart: Option<WriteMultipart>,
}

impl ObjectStoreWriter {
    pub fn new(multipart: WriteMultipart) -> Self {
        Self {
            multipart: Some(multipart),
        }
    }
}

impl VortexWrite for ObjectStoreWriter {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.multipart
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))
            .map(|mp| mp.write(buffer.as_slice()))?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        Ok(self
            .multipart
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))
            .map(|mp| mp.wait_for_capacity(0))?
            .await?)
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        let mp = mem::take(&mut self.multipart);
        mp.ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))?
            .finish()
            .await?;
        Ok(())
    }
}
