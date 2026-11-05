// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory for creating [`VortexReadAt`] instances for [`PartitionedFile`]s.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::Weak;

use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::object_store::ObjectStoreUrl;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use url::Url;
use vortex::array::memory::MemorySessionExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

/// Factory to create [`VortexReadAt`] instances for a `PartitionedFile`.
///
/// Plug a custom implementation into [`VortexSource::with_vortex_reader_factory`]
/// when the default object-store reader is not sufficient, for example to:
///
/// - reuse an application-level metadata cache,
/// - wrap reads with custom authentication or routing,
/// - coalesce I/O in a remote storage layer.
///
/// [`VortexSource::with_vortex_reader_factory`]: crate::VortexSource::with_vortex_reader_factory
pub trait VortexReaderFactory: Debug + Send + Sync + 'static {
    /// Create a reader for a target object.
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>>;
}

/// Default [`VortexReaderFactory`] backed by DataFusion's [`ObjectStore`].
///
/// This is the reader used by [`crate::VortexSource`] and
/// [`crate::VortexFormat`] unless a
/// custom factory is supplied. It works with any object store that DataFusion
/// has registered for the scan. With [`crate::VortexTableOptions::scan_v2`], files of DataFusion's
/// local file system store are read natively instead, keeping each file open while it is scanned.
#[derive(Debug)]
pub struct DefaultVortexReaderFactory {
    object_store: Arc<dyn ObjectStore>,
    local: Option<LocalFileReaders>,
}

/// Native readers for the files of a local object store.
///
/// `LocalFileSystem` opens and stats the file in a blocking task on every ranged read, and
/// `ObjectStoreReadAt` is tuned for remote storage: 192 concurrent requests coalesced up to 16MB.
/// A [`FileReadAt`] keeps the file open and coalesces as tuned for local disk.
pub(crate) struct LocalFileReaders {
    /// The local store, which resolves an object path to a file path, prefix included.
    store: LocalFileSystem,
    /// The reader of each file, shared while any partition of the scan reads the file.
    readers: Arc<DashMap<Path, Weak<FileReadAt>>>,
}

impl Debug for LocalFileReaders {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalFileReaders")
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

impl LocalFileReaders {
    /// Native readers for `object_store` when it is the local file system registered for
    /// `object_store_url`.
    ///
    /// `ObjectStore` cannot be downcast, so the store is recognised by its `Display`, which for
    /// `LocalFileSystem` is its root URL. Any other store, including one wrapping a local store,
    /// keeps reading through the object store.
    pub(crate) fn try_new(
        object_store_url: &ObjectStoreUrl,
        object_store: &dyn ObjectStore,
        readers: Arc<DashMap<Path, Weak<FileReadAt>>>,
    ) -> Option<Self> {
        let url: &Url = object_store_url.as_ref();
        if url.scheme() != "file" {
            return None;
        }
        let display = object_store.to_string();
        let root = display
            .strip_prefix("LocalFileSystem(")?
            .strip_suffix(')')
            .and_then(|root| Url::parse(root).ok())
            .filter(|root| root.scheme() == "file")?;
        let store = if root.path() == "/" {
            LocalFileSystem::new()
        } else {
            LocalFileSystem::new_with_prefix(root.to_file_path().ok()?).ok()?
        };
        Some(Self { store, readers })
    }

    /// The reader of the file at `location`, opening it unless a partition still holds it.
    fn reader(&self, location: &Path, session: &VortexSession) -> VortexResult<Arc<FileReadAt>> {
        match self.readers.entry(location.clone()) {
            Entry::Occupied(mut entry) => {
                if let Some(reader) = entry.get().upgrade() {
                    return Ok(reader);
                }
                let reader = self.open(location, session)?;
                entry.insert(Arc::downgrade(&reader));
                Ok(reader)
            }
            Entry::Vacant(entry) => {
                let reader = self.open(location, session)?;
                entry.insert(Arc::downgrade(&reader));
                Ok(reader)
            }
        }
    }

    fn open(&self, location: &Path, session: &VortexSession) -> VortexResult<Arc<FileReadAt>> {
        let path = self
            .store
            .path_to_filesystem(location)
            .map_err(|e| vortex_err!("Invalid local path {location}: {e}"))?;
        Ok(Arc::new(FileReadAt::open_with_allocator(
            path,
            session.handle(),
            session.allocator(),
        )?))
    }
}

impl DefaultVortexReaderFactory {
    /// Creates a new factory from an [`ObjectStore`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use object_store::memory::InMemory;
    /// use vortex_datafusion::reader::DefaultVortexReaderFactory;
    ///
    /// let factory = DefaultVortexReaderFactory::new(Arc::new(InMemory::new()));
    /// # let _ = factory;
    /// ```
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            object_store,
            local: None,
        }
    }

    /// Reads files of a local object store natively, see [`LocalFileReaders`].
    pub(crate) fn with_local_file_readers(mut self, local: Option<LocalFileReaders>) -> Self {
        self.local = local;
        self
    }
}

impl VortexReaderFactory for DefaultVortexReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        if let Some(local) = &self.local {
            match local.reader(file.path(), session) {
                Ok(reader) => return Ok(reader as _),
                // The object store reports the failure, if any, as it would without local reads.
                Err(e) => tracing::debug!(
                    "Falling back to the object store to read {}: {e}",
                    file.path()
                ),
            }
        }
        Ok(Arc::new(ObjectStoreReadAt::new_with_allocator(
            Arc::clone(&self.object_store),
            file.path().clone(),
            session.handle(),
            session.allocator(),
        )) as _)
    }
}

#[cfg(test)]
mod tests {
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::buffer::Alignment;
    use vortex::io::CoalesceConfig;

    use super::*;

    fn local_readers(url: &str, store: &dyn ObjectStore) -> VortexResult<Option<LocalFileReaders>> {
        let url = ObjectStoreUrl::parse(url).map_err(|e| vortex_err!("{e}"))?;
        Ok(LocalFileReaders::try_new(
            &url,
            store,
            Arc::new(DashMap::default()),
        ))
    }

    #[rstest]
    #[case::local("file://", Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>, true)]
    #[case::memory("file://", Arc::new(InMemory::new()) as Arc<dyn ObjectStore>, false)]
    #[case::remote_scheme("s3://bucket", Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>, false)]
    fn test_local_file_readers_detects_local_store(
        #[case] url: &str,
        #[case] store: Arc<dyn ObjectStore>,
        #[case] local: bool,
    ) -> VortexResult<()> {
        assert_eq!(local_readers(url, store.as_ref())?.is_some(), local);
        Ok(())
    }

    #[tokio::test]
    async fn test_local_file_readers_resolve_prefix_and_share_handle() -> VortexResult<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join("nested"))?;
        let contents = (0..=255_u8).collect::<Vec<_>>();
        std::fs::write(dir.path().join("nested").join("data.vortex"), &contents)?;

        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).map_err(|e| vortex_err!("{e}"))?);
        let local = local_readers("file://", store.as_ref())?;
        assert!(local.is_some());
        let factory = DefaultVortexReaderFactory::new(store).with_local_file_readers(local);
        let session = VortexSession::default();
        let file = PartitionedFile::new("nested/data.vortex", contents.len() as u64);

        let reader = factory
            .create_reader(&file, &session)
            .map_err(|e| vortex_err!("{e}"))?;
        let expected_uri = std::fs::canonicalize(dir.path().join("nested").join("data.vortex"))?;
        assert_eq!(reader.uri().map(|uri| uri.as_ref()), expected_uri.to_str());
        assert_eq!(
            reader.coalesce_config().map(|config| config.max_size),
            Some(CoalesceConfig::file().max_size)
        );
        let bytes = reader.read_at(10, 20, Alignment::none()).await?;
        assert_eq!(bytes.to_host().await.as_slice(), &contents[10..30]);

        // Partitions reading the file at the same time share its open handle.
        let again = factory
            .create_reader(&file, &session)
            .map_err(|e| vortex_err!("{e}"))?;
        assert!(Arc::ptr_eq(&reader, &again));
        Ok(())
    }
}
