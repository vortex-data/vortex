// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::common::Result as DFResult;
use datafusion::datasource::listing::PartitionedFile;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use vortex::array::buffer::BufferHandle;
use vortex::array::memory::MemorySessionExt;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::ReadAtRequest;
use vortex::io::ReadAtStream;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::reader::VortexReaderFactory;

#[derive(Debug)]
pub(super) struct ReaderFactory(pub Arc<dyn ObjectStore>);

impl VortexReaderFactory for ReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(Uncoalesced(
            ObjectStoreReadAt::new_with_allocator(
                Arc::clone(&self.0),
                file.path().clone(),
                session.handle(),
                session.allocator(),
            ),
        )))
    }
}

// The default None coalesce_config makes physical reads independent of which neighboring
// segment futures happen to have registered when the background I/O driver is polled.
struct Uncoalesced(ObjectStoreReadAt);

impl VortexReadAt for Uncoalesced {
    fn uri(&self) -> Option<&Arc<str>> {
        self.0.uri()
    }
    fn concurrency(&self) -> usize {
        self.0.concurrency()
    }
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.0.size()
    }
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.0.read_at(offset, length, alignment)
    }
    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        self.0.read_ranges(requests)
    }
}
