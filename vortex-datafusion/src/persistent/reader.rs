// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Factory for creating [`VortexReadAt`][vortex::io::VortexReadAt] instances
//! from [`PartitionedFile`]s.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use object_store::ObjectStore;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// Factory to create [`VortexReadAt`] instances to read the target file.
pub trait VortexReaderFactory: Debug + Send + Sync + 'static {
    /// Create a reader for a target object.
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>>;
}

/// Default factory, creates [`ObjectStore`] backed readers for files,
/// works with multiple cloud providers.
#[derive(Debug)]
pub struct DefaultVortexReaderFactory {
    object_store: Arc<dyn ObjectStore>,
}

impl DefaultVortexReaderFactory {
    /// Creates new instance
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
        Ok(Arc::new(ObjectStoreReadAt::new(
            Arc::clone(&self.object_store),
            file.path().as_ref().into(),
            session.handle(),
        )) as _)
    }
}
