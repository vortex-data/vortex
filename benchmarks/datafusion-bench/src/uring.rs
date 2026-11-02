// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Local-file completion I/O benchmark configuration and native I/O counters.

use std::fmt;
use std::sync::Arc;

use datafusion::datasource::listing::PartitionedFile;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use parking_lot::Mutex;
use tokio::sync::Semaphore;
use vortex::array::memory::MemorySessionExt;
use vortex::io::VortexReadAt;
use vortex::io::object_store::ObjectStoreReadAt;
use vortex::io::read_at2::FutureReadAt2;
use vortex::io::read_at2::ReadAt2IoService;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::io::std_file::FileReadAt;
use vortex::io::uring::UringStats;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_bench::uring::LocalUring;
use vortex_datafusion::reader::ScanDriver;
use vortex_datafusion::reader::VortexReaderFactory;

static FACTORY: Mutex<Option<Arc<LocalUringFactory>>> = Mutex::new(None);

/// Returns the configured completion factory, or `None` for the existing reader.
pub fn factory(scan_v2: bool) -> anyhow::Result<Option<Arc<dyn VortexReaderFactory>>> {
    let backend = std::env::var("VORTEX_IO_BACKEND").unwrap_or_default();
    factory_for_backend(scan_v2, &backend)
}

fn factory_for_backend(
    scan_v2: bool,
    backend: &str,
) -> anyhow::Result<Option<Arc<dyn VortexReaderFactory>>> {
    if !scan_v2 || backend.is_empty() || backend == "legacy" {
        return Ok(None);
    }
    if backend == "memory" || backend == "memory-per-core" {
        return Ok(Some(crate::memory::factory()?));
    }
    if backend == "future-file" || backend == "future-object-store" {
        return Ok(Some(Arc::new(FutureReaderFactory {
            runtime: TokioRuntime::new(tokio::runtime::Handle::current()),
            object_store: (backend == "future-object-store")
                .then(|| Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>),
            files: Mutex::default(),
        })));
    }
    let mut factory = FACTORY.lock();
    if let Some(factory) = factory.as_ref() {
        return Ok(Some(Arc::clone(factory) as _));
    }
    let io = LocalUring::from_env(backend)?;
    let active_chunks = std::env::var("VORTEX_SCAN_V2_MAX_ACTIVE_CHUNKS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if let Some(active_chunks) = active_chunks {
        anyhow::ensure!(active_chunks > 0, "Active scan chunks must be positive");
        anyhow::ensure!(
            io.workers().is_none(),
            "Native scan workers control their own admission"
        );
    }
    let value = Arc::new(LocalUringFactory {
        io,
        chunk_admission: active_chunks.map(|limit| Arc::new(Semaphore::new(limit))),
        admit_after_io: std::env::var("VORTEX_SCAN_V2_ADMIT_AFTER_IO")
            .is_ok_and(|value| value == "1"),
        runtime: TokioRuntime::new(tokio::runtime::Handle::current()),
    });
    *factory = Some(Arc::clone(&value));
    Ok(Some(value))
}

struct LocalUringFactory {
    io: LocalUring,
    chunk_admission: Option<Arc<Semaphore>>,
    admit_after_io: bool,
    runtime: TokioRuntime,
}

struct FutureReaderFactory {
    runtime: TokioRuntime,
    object_store: Option<Arc<dyn ObjectStore>>,
    files: Mutex<HashMap<Path, Arc<dyn VortexReadAt>>>,
}

impl fmt::Debug for FutureReaderFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FutureReaderFactory")
            .field("object_store", &self.object_store.is_some())
            .finish_non_exhaustive()
    }
}

impl VortexReaderFactory for FutureReaderFactory {
    fn create_reader(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        let mut files = self.files.lock();
        if let Some(reader) = files.get(file.path()) {
            return Ok(Arc::clone(reader));
        }
        let reader: Arc<dyn VortexReadAt> = match &self.object_store {
            Some(store) => Arc::new(ObjectStoreReadAt::new_with_allocator(
                Arc::clone(store),
                file.path().clone(),
                self.runtime.handle(),
                session.allocator(),
            )),
            None => {
                let path = LocalFileSystem::new().path_to_filesystem(file.path())?;
                Arc::new(
                    FileReadAt::open_with_allocator(
                        path,
                        self.runtime.handle(),
                        session.allocator(),
                    )
                    .map_err(|error| exec_datafusion_err!("Opening future reader: {error}"))?,
                )
            }
        };
        files.insert(file.path().clone(), Arc::clone(&reader));
        Ok(reader)
    }

    fn configure_v2_scan(
        &self,
        file: &PartitionedFile,
        session: &VortexSession,
        scan: ScanBuilder,
    ) -> DFResult<ScanBuilder> {
        let reader = self.create_reader(file, session)?;
        let factory = Arc::new(move || Ok(Arc::new(FutureReadAt2::new(Arc::clone(&reader))) as _));
        Ok(scan.with_io_service(Arc::new(ReadAt2IoService::new(factory))))
    }
}

impl fmt::Debug for LocalUringFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalUringFactory").finish_non_exhaustive()
    }
}

impl VortexReaderFactory for LocalUringFactory {
    fn configure_v2_driver(&self, driver: ScanDriver) -> ScanDriver {
        match self.io.workers() {
            Some(workers) => driver.with_uring_workers(workers.clone()),
            None => match &self.chunk_admission {
                Some(admission) => {
                    driver.with_chunk_admission(Arc::clone(admission), self.admit_after_io)
                }
                None => driver,
            },
        }
    }

    fn create_reader(
        &self,
        file: &PartitionedFile,
        _session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        let path = LocalFileSystem::new().path_to_filesystem(file.path())?;
        Ok(Arc::new(
            FileReadAt::open(path, self.runtime.handle())
                .map_err(|error| exec_datafusion_err!("Opening metadata reader: {error}"))?,
        ))
    }

    fn configure_v2_scan(
        &self,
        file: &PartitionedFile,
        _session: &VortexSession,
        scan: ScanBuilder,
    ) -> DFResult<ScanBuilder> {
        let path = LocalFileSystem::new().path_to_filesystem(file.path())?;
        let reader = self
            .io
            .reader(&path)
            .map_err(|error| exec_datafusion_err!("Opening native reader: {error}"))?;
        Ok(scan.with_io_service(Arc::new(ReadAt2IoService::new(reader))))
    }
}

/// Counts submitted reads across all files of the process's native reader.
pub fn snapshot() -> UringStats {
    FACTORY
        .lock()
        .as_ref()
        .map_or_else(UringStats::default, |factory| factory.io.snapshot())
}

/// Releases reusable mappings before cold-cache eviction, preserving outstanding buffer owners.
pub fn release_mappings() -> usize {
    FACTORY
        .lock()
        .as_ref()
        .map_or(0, |factory| factory.io.release_mappings())
}

#[cfg(test)]
mod tests {
    use super::factory_for_backend;

    #[test]
    fn v1_keeps_its_reader_when_completion_backends_are_configured() -> anyhow::Result<()> {
        for backend in [
            "shared-1-mapped",
            "shared-2-mapped",
            "per-core-mapped",
            "future-file",
            "memory",
        ] {
            assert!(factory_for_backend(false, backend)?.is_none(), "{backend}");
        }
        Ok(())
    }
}
