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
use vortex_error::{vortex_panic, VortexError, VortexResult};

use crate::{VortexBufReader, VortexReadAt, VortexWrite, BUFFER_ALIGNMENT};

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

            // Allocate a Vec<u8> that has at least BUFFER_ALIGNMENT-1 bytes of extra capacity.
            // This is important as it allows us to return a properly aligned Bytes to the caller,
            // without needing to write a custom Allocator.
            let alloc_size =
                ((len as usize) + (BUFFER_ALIGNMENT - 1)).next_multiple_of(BUFFER_ALIGNMENT);
            let mut buf = Vec::<u8>::with_capacity(alloc_size as _);
            let padding = buf.as_ptr().align_offset(BUFFER_ALIGNMENT);
            unsafe { buf.set_len(padding) };

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

            // bytes_unaligned will contain the entire allocation, so that on Drop the entire buf
            // is freed.
            //
            // bytes_aligned is a sliced view on top of bytes_unaligned.
            //
            // bytes_aligned
            //     | parent    \  *ptr
            //     v            |
            // bytes_unaligned  |
            //     |            |
            //     | *ptr       |
            //     v            v
            //     +------------+------------------+----------------+
            //     | padding    |   content        | spare capacity |
            //     +------------+------------------+----------------+
            //
            let bytes_unaligned = Bytes::from(buf);
            let bytes_aligned = bytes_unaligned.slice(padding..);

            assert_eq!(
                bytes_aligned.as_ptr().align_offset(BUFFER_ALIGNMENT),
                0,
                "buffer must be 64-byte aligned"
            );
            Ok(bytes_aligned)
        })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    fn size(&self) -> impl Future<Output = u64> + 'static {
        let object_store = self.object_store.clone();
        let location = self.location.clone();

        Box::pin(async move {
            object_store
                .head(&location)
                .await
                .map_err(VortexError::ObjectStore)
                .unwrap_or_else(|err| {
                    vortex_panic!(err, "Failed to get size of object at location {}", location)
                })
                .size as u64
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
