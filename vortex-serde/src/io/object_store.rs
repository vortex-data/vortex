#![cfg(feature = "object_store")]

use std::future::Future;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;
use std::{io, mem};

use bytes::BytesMut;
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreScheme, WriteMultipart};
use url::Url;
use vortex_buffer::io_buf::IoBuf;
use vortex_buffer::Buffer;
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexResult};

use crate::io::{VortexRead, VortexReadAt, VortexWrite};

pub trait ObjectStoreExt {
    fn vortex_read(
        &self,
        location: &Path,
        range: Range<usize>,
    ) -> impl Future<Output = VortexResult<impl VortexRead>>;

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
    ) -> VortexResult<impl VortexRead> {
        let bytes = self.get_range(location, range).await?;
        Ok(Cursor::new(Buffer::Bytes(bytes)))
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

    fn better_parse_url(url_str: &str) -> VortexResult<(Box<dyn ObjectStore>, Path)> {
        let url = Url::parse(url_str)?;

        let (scheme, path) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
        let store: Box<dyn ObjectStore> = match scheme {
            ObjectStoreScheme::Local => Box::new(LocalFileSystem::default()),
            ObjectStoreScheme::AmazonS3 => {
                Box::new(AmazonS3Builder::from_env().with_url(url_str).build()?)
            }
            ObjectStoreScheme::GoogleCloudStorage => Box::new(
                GoogleCloudStorageBuilder::from_env()
                    .with_url(url_str)
                    .build()?,
            ),
            ObjectStoreScheme::MicrosoftAzure => Box::new(
                MicrosoftAzureBuilder::from_env()
                    .with_url(url_str)
                    .build()?,
            ),
            ObjectStoreScheme::Http => Box::new(
                HttpBuilder::new()
                    .with_url(&url[..url::Position::BeforePath])
                    .build()?,
            ),
            otherwise => vortex_bail!("unrecognized object store scheme: {:?}", otherwise),
        };

        Ok((store, path))
    }

    pub async fn try_new_from_url(url: &str) -> VortexResult<Self> {
        let (object_store, location) = Self::better_parse_url(url)?;
        Ok(Self::new(Arc::from(object_store), location))
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    async fn read_at_into(&self, pos: u64, mut buffer: BytesMut) -> io::Result<BytesMut> {
        let start_range = pos as usize;
        let bytes = self
            .object_store
            .get_range(&self.location, start_range..(start_range + buffer.len()))
            .await?;
        buffer.as_mut().copy_from_slice(bytes.as_ref());
        Ok(buffer)
    }

    async fn size(&self) -> u64 {
        self.object_store
            .head(&self.location)
            .await
            .map_err(VortexError::ObjectStore)
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to get size of object at location {}",
                    self.location
                )
            })
            .size as u64
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
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> std::io::Result<B> {
        self.multipart
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))
            .map(|mp| mp.write(buffer.as_slice()))?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        Ok(self
            .multipart
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))
            .map(|mp| mp.wait_for_capacity(0))?
            .await?)
    }

    async fn shutdown(&mut self) -> std::io::Result<()> {
        let mp = mem::take(&mut self.multipart);
        mp.ok_or_else(|| io::Error::new(io::ErrorKind::Other, "multipart already finished"))?
            .finish()
            .await?;
        Ok(())
    }
}
