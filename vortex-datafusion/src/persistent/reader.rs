use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result as DFResult};
use object_store::ObjectStore;
use vortex::io::file::ReadSource;
use vortex::io::file::object_store::ObjectStoreReadSource;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

#[async_trait]
pub trait VortexReaderFactory: Debug + Send + Sync + 'static {
    async fn create_reader(
        &self,
        path: &str,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn ReadSource>>;
}

#[derive(Debug)]
pub struct DefaultVortexReaderFactory {
    object_store: Arc<dyn ObjectStore>,
}

impl DefaultVortexReaderFactory {
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self { object_store }
    }
}

#[async_trait]
impl VortexReaderFactory for DefaultVortexReaderFactory {
    async fn create_reader(
        &self,
        path: &str,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn ReadSource>> {
        ObjectStoreReadSource::new(self.object_store.clone(), path.into())
            .into_read_source(session.handle())
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}
