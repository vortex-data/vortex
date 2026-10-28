// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory for creating [`VortexReadAt`] instances for [`PartitionedFile`]s.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use object_store::ObjectStore;
use vortex::array::memory::MemorySessionExt;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
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
