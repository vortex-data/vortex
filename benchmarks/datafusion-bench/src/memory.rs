// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preloaded compressed bytes for measuring scan execution without storage reads.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::OnceLock;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::time::Instant;

use datafusion::datasource::listing::PartitionedFile;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use object_store::path::Path;
use parking_lot::Mutex;
use url::Url;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::read_at2::ReadAt2IoService;
use vortex::io::read_at2::ReadCompletion;
use vortex::io::read_at2::ReadId;
use vortex::io::read_at2::ReadRequest;
use vortex::io::read_at2::ReadResult;
use vortex::io::read_at2::ReadTarget;
use vortex::io::read_at2::VortexReadAt2;
use vortex::io::uring::UringOptions;
use vortex::io::uring::UringWorkerPool;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_bench::Format;
use vortex_datafusion::reader::ScanDriver;
use vortex_datafusion::reader::VortexReaderFactory;

static FACTORY: OnceLock<Arc<MemoryFactory>> = OnceLock::new();

/// Loads the selected input format before any query timers or table registration.
pub fn prepare(base: &Url, formats: &[Format], cold: bool) -> anyhow::Result<()> {
    let backend = std::env::var("VORTEX_IO_BACKEND").unwrap_or_default();
    if backend != "memory" && backend != "memory-per-core" {
        return Ok(());
    }
    anyhow::ensure!(
        !cold,
        "Preloaded memory cannot be labelled a cold-storage run"
    );
    let base = base
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Memory input requires local files"))?;
    let started = Instant::now();
    let mut files = HashMap::default();
    for format in formats {
        anyhow::ensure!(
            *format == Format::OnDiskVortex,
            "Memory input requires --formats vortex"
        );
        preload(&base.join(format.name()), &mut files)?;
    }
    anyhow::ensure!(!files.is_empty(), "No Vortex files found to preload");
    let bytes: usize = files.values().map(ByteBuffer::len).sum();
    eprintln!(
        "memory_preload {}",
        serde_json::json!({
            "files": files.len(), "bytes": bytes, "seconds": started.elapsed().as_secs_f64(),
        })
    );
    let workers = if backend == "memory-per-core" {
        let count = std::env::var("VORTEX_IO_WORKERS")
            .unwrap_or_else(|_| "32".into())
            .parse()?;
        Some(UringWorkerPool::new(count, UringOptions::default())?)
    } else {
        None
    };
    FACTORY
        .set(Arc::new(MemoryFactory { files, workers }))
        .map_err(|_| anyhow::anyhow!("Memory input already initialized"))?;
    Ok(())
}

fn preload(directory: &FsPath, files: &mut HashMap<Path, ByteBuffer>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            preload(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "vortex") {
            let mut file = File::open(&path)?;
            let length = usize::try_from(file.metadata()?.len())?;
            let mut bytes = ByteBufferMut::zeroed_aligned(length, Alignment::new(4096));
            file.read_exact(bytes.as_mut())?;
            files.insert(Path::from_filesystem_path(&path)?, bytes.freeze());
        }
    }
    Ok(())
}

/// Returns the prepared memory reader shared by every table in this process.
pub fn factory() -> anyhow::Result<Arc<dyn VortexReaderFactory>> {
    let factory = FACTORY
        .get()
        .ok_or_else(|| anyhow::anyhow!("Memory input was not preloaded"))?;
    Ok(Arc::clone(factory) as _)
}

struct MemoryFactory {
    files: HashMap<Path, ByteBuffer>,
    workers: Option<UringWorkerPool>,
}

impl std::fmt::Debug for MemoryFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryFactory")
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl MemoryFactory {
    fn bytes(&self, file: &PartitionedFile) -> DFResult<ByteBuffer> {
        self.files
            .get(file.path())
            .cloned()
            .ok_or_else(|| exec_datafusion_err!("File was not preloaded: {}", file.path()))
    }
}

impl VortexReaderFactory for MemoryFactory {
    fn configure_v2_driver(&self, driver: ScanDriver) -> ScanDriver {
        match &self.workers {
            Some(workers) => driver.with_uring_workers(workers.clone()),
            None => driver,
        }
    }

    fn create_reader(
        &self,
        file: &PartitionedFile,
        _session: &VortexSession,
    ) -> DFResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(self.bytes(file)?))
    }

    fn configure_v2_scan(
        &self,
        file: &PartitionedFile,
        _session: &VortexSession,
        scan: ScanBuilder,
    ) -> DFResult<ScanBuilder> {
        let bytes = self.bytes(file)?;
        let factory = Arc::new(move || {
            Ok(Arc::new(MemoryReadAt2 {
                bytes: bytes.clone(),
                state: Mutex::default(),
            }) as _)
        });
        Ok(scan.with_io_service(Arc::new(ReadAt2IoService::new(factory))))
    }
}

struct MemoryReadAt2 {
    bytes: ByteBuffer,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    queued: VecDeque<ReadRequest>,
    waker: Option<Waker>,
}

impl MemoryReadAt2 {
    fn read(&self, target: ReadTarget) -> VortexResult<ReadResult> {
        match target {
            ReadTarget::Size => Ok(ReadResult::Size(self.bytes.len() as u64)),
            ReadTarget::Range {
                offset,
                length,
                alignment,
            } => {
                let start = usize::try_from(offset)?;
                let end = start
                    .checked_add(length)
                    .ok_or_else(|| vortex_err!("Read range overflow"))?;
                if end > self.bytes.len() {
                    vortex_bail!("Read range exceeds preloaded file");
                }
                Ok(ReadResult::Bytes(BufferHandle::new_host(
                    self.bytes.slice_unaligned(start..end).aligned(alignment),
                )))
            }
            ReadTarget::Suffix { length, alignment } => {
                let start = self.bytes.len().saturating_sub(length);
                Ok(ReadResult::Suffix {
                    size: self.bytes.len() as u64,
                    bytes: BufferHandle::new_host(
                        self.bytes.slice_unaligned(start..).aligned(alignment),
                    ),
                })
            }
        }
    }
}

impl VortexReadAt2 for MemoryReadAt2 {
    fn submit(&self, requests: &[ReadRequest]) -> VortexResult<()> {
        let mut state = self.state.lock();
        let mut ids: HashSet<_> = state.queued.iter().map(|request| request.id).collect();
        for request in requests {
            if !ids.insert(request.id) {
                vortex_bail!("Duplicate memory read identity {:?}", request.id);
            }
        }
        state.queued.extend(requests.iter().copied());
        let waker = state.waker.take();
        drop(state);
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    fn try_complete(&self) -> VortexResult<Option<ReadCompletion>> {
        let request = self.state.lock().queued.pop_front();
        Ok(request.map(|request| ReadCompletion {
            id: request.id,
            result: self.read(request.target),
        }))
    }

    fn poll_complete(&self, cx: &mut Context<'_>) -> Poll<VortexResult<ReadCompletion>> {
        let mut state = self.state.lock();
        match state.queued.pop_front() {
            Some(request) => {
                drop(state);
                Poll::Ready(Ok(ReadCompletion {
                    id: request.id,
                    result: self.read(request.target),
                }))
            }
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    fn wait(&self) -> VortexResult<ReadCompletion> {
        self.try_complete()?
            .ok_or_else(|| vortex_err!("Waiting with no memory read outstanding"))
    }

    fn cancel(&self, id: ReadId) {
        self.state.lock().queued.retain(|request| request.id != id);
    }
}
