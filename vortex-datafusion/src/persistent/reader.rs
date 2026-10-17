// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory for creating [`VortexReadAt`] instances for [`PartitionedFile`]s.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use vortex::array::memory::MemorySessionExt;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::MmapReadAt;
use vortex::session::VortexSession;

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
/// has registered for the scan.
#[derive(Debug)]
pub struct DefaultVortexReaderFactory {
    object_store: Arc<dyn ObjectStore>,
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
        Self { object_store }
    }
}

impl VortexReaderFactory for DefaultVortexReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(ObjectStoreReadAt::new_with_allocator(
            Arc::clone(&self.object_store),
            file.path().clone(),
            session.handle(),
            session.allocator(),
        )) as _)
    }
}

/// [`VortexReaderFactory`] that reads local files through memory maps of them.
///
/// Reads return slices of the map instead of copying each segment out of the page cache into a
/// freshly allocated buffer on a blocking thread, which is a large part of a scan's cost when the
/// files are cached in memory.
#[derive(Debug)]
pub struct MmapVortexReaderFactory {
    local: Arc<LocalFileSystem>,
}

impl MmapVortexReaderFactory {
    /// Creates a factory that maps files of `local`.
    ///
    /// # Safety
    ///
    /// No file read through this factory may be truncated or modified while a buffer read from it
    /// is alive, see [`MmapReadAt::open`].
    pub unsafe fn new(local: Arc<LocalFileSystem>) -> Self {
        Self { local }
    }
}

impl VortexReaderFactory for MmapVortexReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        _session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        let path = self
            .local
            .path_to_filesystem(file.path())
            .map_err(|e| DataFusionError::ObjectStore(Box::new(e)))?;
        // SAFETY: `new` requires that the files do not change while they are mapped.
        let reader = unsafe { MmapReadAt::open(path) }
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Arc::new(reader) as _)
    }
}
