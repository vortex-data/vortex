// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared local-file completion I/O configuration for SQL benchmarks.

use std::path::Path;
use std::path::PathBuf;

use parking_lot::Mutex;
use vortex::io::read_at2::ReadAt2Factory;
use vortex::io::uring::FileMode;
use vortex::io::uring::SharedUringPool;
use vortex::io::uring::UringFile;
use vortex::io::uring::UringOptions;
use vortex::io::uring::UringStats;
use vortex::io::uring::UringWorkerPool;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::parallelism::get_available_parallelism;

/// Native readers and reactors shared by the files in a benchmark process.
pub struct LocalUring {
    mode: FileMode,
    mapped: bool,
    mapped_random_access: bool,
    execution: Execution,
    files: Mutex<HashMap<PathBuf, UringFile>>,
}

enum Execution {
    PerCore(UringWorkerPool),
    Shared(SharedUringPool),
}

impl LocalUring {
    /// Constructs the named backend with the benchmark's `VORTEX_IO_*` settings.
    pub fn from_env(backend: &str) -> anyhow::Result<Self> {
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
        let execution = match rings {
            Some(rings) => Execution::Shared(
                if per_core_reactors && std::env::var("VORTEX_IO_PIN_WORKERS").as_deref() != Ok("0")
                {
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
        Ok(Self {
            mode,
            mapped,
            mapped_random_access: std::env::var("VORTEX_IO_MAPPED_RANDOM_ACCESS")
                .is_ok_and(|value| value == "1"),
            execution,
            files: Mutex::default(),
        })
    }

    /// Dedicated compute workers, if this backend requires them to drive its rings.
    pub fn workers(&self) -> Option<&UringWorkerPool> {
        match &self.execution {
            Execution::PerCore(workers) => Some(workers),
            Execution::Shared(_) => None,
        }
    }

    /// Creates read scopes for an immutable benchmark input, reusing its descriptor and mapping.
    pub fn reader(&self, path: &Path) -> anyhow::Result<ReadAt2Factory> {
        let mut files = self.files.lock();
        let native = if let Some(native) = files.get(path) {
            native.clone()
        } else {
            let native = if self.mapped {
                // SAFETY: benchmark inputs are immutable after ingestion. Results and temporary
                // spill files are written separately and never opened by this reader.
                unsafe {
                    if self.mapped_random_access {
                        UringFile::open_mapped_random_access(path)
                    } else {
                        UringFile::open_mapped(path)
                    }
                }
            } else {
                UringFile::open(path, self.mode)
            }?;
            files.insert(path.to_path_buf(), native.clone());
            native
        };
        Ok(match &self.execution {
            Execution::PerCore(_) => native.factory(),
            Execution::Shared(reactors) => reactors.factory(native),
        })
    }

    /// Counts submitted reads across every file opened by this backend.
    pub fn snapshot(&self) -> UringStats {
        let mut result = UringStats::default();
        for file in self.files.lock().values() {
            let stats = file.stats();
            result.logical_reads += stats.logical_reads;
            result.logical_bytes += stats.logical_bytes;
            result.physical_reads += stats.physical_reads;
            result.physical_bytes += stats.physical_bytes;
            result.resident_reads += stats.resident_reads;
            result.resident_bytes += stats.resident_bytes;
        }
        result
    }

    /// Releases reusable mappings before eviction, preserving outstanding buffer owners.
    pub fn release_mappings(&self) -> usize {
        self.files
            .lock()
            .values()
            .filter(|file| file.release_mapping())
            .count()
    }
}
