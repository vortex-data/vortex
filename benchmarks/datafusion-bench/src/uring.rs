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
use vortex::io::uring::FileMode;
use vortex::io::uring::SharedUringPool;
use vortex::io::uring::UringFile;
use vortex::io::uring::UringOptions;
use vortex::io::uring::UringStats;
use vortex::io::uring::UringWorkerPool;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::parallelism::get_available_parallelism;
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
    let workers = std::env::var("VORTEX_IO_WORKERS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or_else(|| get_available_parallelism().unwrap_or(1));
    let per_core_reactors = backend.starts_with("per-core-reactor-");
    let (mode, rings, mapped) = match backend.strip_suffix("-unmerged").unwrap_or(backend) {
        "per-core-buffered" => (FileMode::Buffered, None, false),
        "per-core-direct" => (FileMode::Direct, None, false),
        "per-core-reactor-buffered" => (FileMode::Buffered, Some(workers), false),
        "per-core-reactor-direct" => (FileMode::Direct, Some(workers), false),
        "per-core-reactor-mapped" => (FileMode::Buffered, Some(workers), true),
        "shared-1-buffered" => (FileMode::Buffered, Some(1), false),
        "shared-2-buffered" => (FileMode::Buffered, Some(2), false),
        "shared-1-direct" => (FileMode::Direct, Some(1), false),
        "shared-2-direct" => (FileMode::Direct, Some(2), false),
        "per-core-mapped" => (FileMode::Buffered, None, true),
        "shared-1-mapped" => (FileMode::Buffered, Some(1), true),
        "shared-2-mapped" => (FileMode::Buffered, Some(2), true),
        _ => anyhow::bail!("Unknown VORTEX_IO_BACKEND {backend}"),
    };
    let queue_depth = std::env::var("VORTEX_IO_QUEUE_DEPTH")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(64);
    let options = UringOptions {
        queue_depth,
        max_inflight_bytes: std::env::var("VORTEX_IO_MAX_INFLIGHT_BYTES")
            .ok()
            .map(|value| value.parse())
            .transpose()?,
        coalesce: !backend.ends_with("unmerged"),
        max_coalescing_gap: std::env::var("VORTEX_IO_COALESCE_GAP")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0),
        max_coalesced_bytes: std::env::var("VORTEX_IO_COALESCE_MAX_BYTES")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(1 << 20),
        preserve_queue_order: std::env::var("VORTEX_IO_COALESCE_FIFO")
            .is_ok_and(|value| value == "1"),
        coalesce_lookahead: std::env::var("VORTEX_IO_COALESCE_LOOKAHEAD")
            .ok()
            .map(|value| value.parse())
            .transpose()?,
        reuse_inflight: std::env::var("VORTEX_IO_REUSE_INFLIGHT").map_or_else(
            |_| rings.is_some() && !backend.ends_with("unmerged"),
            |value| value == "1",
        ),
        reuse_live_buffers: std::env::var("VORTEX_IO_REUSE_LIVE_BUFFERS")
            .is_ok_and(|value| value == "1"),
        reuse_across_workers: std::env::var("VORTEX_IO_REUSE_ACROSS_WORKERS")
            .is_ok_and(|value| value == "1"),
        check_resident_pages: std::env::var("VORTEX_IO_CHECK_RESIDENT_PAGES")
            .is_ok_and(|value| value == "1"),
        check_queued_resident_pages: std::env::var("VORTEX_IO_CHECK_QUEUED_RESIDENT_PAGES")
            .is_ok_and(|value| value == "1"),
        read_mapped_misses: std::env::var("VORTEX_IO_READ_MAPPED_MISSES")
            .is_ok_and(|value| value == "1"),
        force_async: std::env::var("VORTEX_IO_FORCE_ASYNC").is_ok_and(|value| value == "1"),
        eager_kernel_worker: std::env::var("VORTEX_IO_EAGER_WORKER")
            .is_ok_and(|value| value == "1"),
        max_kernel_workers: std::env::var("VORTEX_IO_MAX_KERNEL_WORKERS")
            .ok()
            .map(|value| value.parse())
            .transpose()?,
        max_active_jobs: std::env::var("VORTEX_IO_ACTIVE_JOBS")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(1),
        max_control_jobs: std::env::var("VORTEX_IO_CONTROL_JOBS")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0),
        control_workers: std::env::var("VORTEX_IO_CONTROL_WORKERS")
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(0),
    };
    let active_chunks = std::env::var("VORTEX_SCAN_V2_MAX_ACTIVE_CHUNKS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if let Some(active_chunks) = active_chunks {
        anyhow::ensure!(active_chunks > 0, "Active scan chunks must be positive");
        anyhow::ensure!(
            rings.is_some(),
            "Native scan workers control their own admission"
        );
    }
    let execution = match rings {
        Some(rings) => Execution::Shared(
            if per_core_reactors && std::env::var("VORTEX_IO_PIN_WORKERS").as_deref() != Ok("0") {
                SharedUringPool::new_pinned(rings, options)?
            } else {
                SharedUringPool::new(rings, options)?
            },
        ),
        None => Execution::PerCore(
            if std::env::var("VORTEX_IO_PIN_WORKERS").as_deref() == Ok("0") {
                UringWorkerPool::with_inherited_affinity(workers, options)?
            } else {
                UringWorkerPool::new(workers, options)?
            },
        ),
    };
    let value = Arc::new(LocalUringFactory {
        mode,
        mapped,
        mapped_random_access: std::env::var("VORTEX_IO_MAPPED_RANDOM_ACCESS")
            .is_ok_and(|value| value == "1"),
        execution,
        chunk_admission: active_chunks.map(|limit| Arc::new(Semaphore::new(limit))),
        admit_after_io: std::env::var("VORTEX_SCAN_V2_ADMIT_AFTER_IO")
            .is_ok_and(|value| value == "1"),
        runtime: TokioRuntime::new(tokio::runtime::Handle::current()),
        files: Mutex::default(),
    });
    *factory = Some(Arc::clone(&value));
    Ok(Some(value))
}

struct LocalUringFactory {
    mode: FileMode,
    mapped: bool,
    mapped_random_access: bool,
    execution: Execution,
    chunk_admission: Option<Arc<Semaphore>>,
    admit_after_io: bool,
    runtime: TokioRuntime,
    files: Mutex<HashMap<Path, UringFile>>,
}

enum Execution {
    PerCore(UringWorkerPool),
    Shared(SharedUringPool),
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
        f.debug_struct("LocalUringFactory")
            .field("mode", &self.mode)
            .field("mapped", &self.mapped)
            .field("mapped_random_access", &self.mapped_random_access)
            .finish_non_exhaustive()
    }
}

impl VortexReaderFactory for LocalUringFactory {
    fn configure_v2_driver(&self, driver: ScanDriver) -> ScanDriver {
        match &self.execution {
            Execution::PerCore(workers) => driver.with_uring_workers(workers.clone()),
            Execution::Shared(_) => match &self.chunk_admission {
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
        let mut files = self.files.lock();
        let native = if let Some(native) = files.get(file.path()) {
            native.clone()
        } else {
            let path = LocalFileSystem::new().path_to_filesystem(file.path())?;
            let native = if self.mapped {
                // SAFETY: benchmark input files are immutable after ingestion. The factory is
                // only used for reads, and generated result files use a separate directory.
                unsafe {
                    if self.mapped_random_access {
                        UringFile::open_mapped_random_access(path)
                    } else {
                        UringFile::open_mapped(path)
                    }
                }
            } else {
                UringFile::open(path, self.mode)
            }
            .map_err(|error| exec_datafusion_err!("Opening native reader: {error}"))?;
            files.insert(file.path().clone(), native.clone());
            native
        };
        let reader = match &self.execution {
            Execution::PerCore(_) => native.factory(),
            Execution::Shared(reactors) => reactors.factory(native),
        };
        Ok(scan.with_io_service(Arc::new(ReadAt2IoService::new(reader))))
    }
}

/// Counts submitted reads across all files of the process's native reader.
pub fn snapshot() -> UringStats {
    let mut result = UringStats::default();
    if let Some(factory) = FACTORY.lock().as_ref() {
        for file in factory.files.lock().values() {
            let stats = file.stats();
            result.logical_reads += stats.logical_reads;
            result.logical_bytes += stats.logical_bytes;
            result.physical_reads += stats.physical_reads;
            result.physical_bytes += stats.physical_bytes;
            result.resident_reads += stats.resident_reads;
            result.resident_bytes += stats.resident_bytes;
        }
    }
    result
}

/// Releases reusable mappings before cold-cache eviction, preserving outstanding buffer owners.
pub fn release_mappings() -> usize {
    FACTORY.lock().as_ref().map_or(0, |factory| {
        factory
            .files
            .lock()
            .values()
            .filter(|file| file.release_mapping())
            .count()
    })
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
