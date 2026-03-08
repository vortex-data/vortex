// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use clap::{Parser, ValueEnum};
use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
use futures::StreamExt;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use vortex::VortexSessionDefault;
use vortex::array::ArrayContext;
use vortex::array::arrays::{ChunkedVTable, StructVTable};
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayParts;
use vortex::buffer::{Alignment, ByteBuffer};
use vortex::dtype::{DType, PType};
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::flat::FlatVTable;
use vortex::layout::segments::SegmentSource;
use vortex::layout::{LayoutChildType, LayoutRef};
use vortex::scan::SplitBy;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap as VortexHashMap;
use vortex_cuda::dynamic_dispatch::{self, DynamicDispatchPlan, build_plan_from_flatbuffer};
use vortex_cuda::layout::{CudaFlatVTable, register_cuda_layout};
use vortex_cuda::{CudaDeviceBuffer, CudaExecutionCtx, CudaSession, CudaSessionExt};

// =====================================================================
// CLI
// =====================================================================

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ScanMode {
    /// Multi-threaded tokio async pipeline: open_buffer → scan → decode
    /// batches on tokio workers → GPU kernels.
    TokioMt,
    /// Same as `tokio-mt` but forced onto a single-threaded tokio runtime.
    /// Isolates Arc contention cost vs inherent Arc cost.
    TokioSt,
    /// Synchronous layout walk + ArrayParts::decode + build_plan.
    /// No async pipeline, no tokio workers — single-threaded decode
    /// using the real ArrayRef construction path.  Backed by mmap.
    Sync,
    /// pread + O_DIRECT cold read into CUDA-pinned buffer, then sync
    /// ArrayRef decode.  No io_uring — plain sequential pread.
    /// Warm iterations reuse the pinned buffer (no I/O).
    Odirect,
    /// io_uring + O_DIRECT cold read into CUDA-pinned buffer, then sync
    /// ArrayRef decode.  Warm iterations reuse the pinned buffer (no I/O).
    IoUringOdirect,
    /// io_uring buffered (page-cached) cold read into CUDA-pinned buffer,
    /// then sync ArrayRef decode.  Warm iterations reuse the pinned buffer.
    IoUringBuffered,
    /// Zero-deserialization: walk layout tree → build plans directly from
    /// flatbuffer metadata → GPU kernels. Bypasses ArrayRef construction.
    Fb,
    /// io_uring + O_DIRECT cold read, then direct-fb decode (no ArrayRef).
    FbIoUringOdirect,
    /// io_uring buffered cold read, then direct-fb decode (no ArrayRef).
    FbIoUringBuffered,
}

#[derive(Parser)]
#[command(
    name = "gpu-scan-bench",
    about = "Benchmark GPU scans of CUDA-compatible Vortex files",
    long_about = "\
Benchmark GPU scans of CUDA-compatible Vortex files.

Cold start input throughput is dominated by the OS read_ahead_kb setting,
which controls how aggressively the kernel prefetches mmap'd pages. The
default (128 KB) is too small to saturate most storage devices.

To increase it for the duration of a benchmark run (replace <dev> with
your block device, e.g. nvme0n1, vda):

    sudo sh -c 'echo 16384 > /sys/block/<dev>/queue/read_ahead_kb' && \\
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && \\
    cargo run --release -p gpu-scan-bench -- file.vortex --iterations 5 --output-size-mb 4001;

Typical results on GH200 with read_ahead_kb=16384:

    Cold:  ~0.07s  (57 GB/s decompressed output)
    Warm:  ~0.02s  (207 GB/s decompressed output)"
)]
struct Cli {
    /// Local path to a .vortex file.
    source: String,

    /// Number of timed scan iterations.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Decompressed output size in MB (for output throughput calculation).
    #[arg(long)]
    output_size_mb: Option<f64>,

    /// Path to write Perfetto trace output.
    #[arg(long)]
    perfetto: Option<PathBuf>,

    /// Scan mode.
    #[arg(long, value_enum, default_value_t = ScanMode::TokioMt)]
    mode: ScanMode,

    /// Output logs as JSON.
    #[arg(long)]
    json: bool,
}

// =====================================================================
// Tracing setup
// =====================================================================

fn init_tracing(json: bool, perfetto_path: Option<&Path>) -> VortexResult<()> {
    let perfetto_layer = perfetto_path
        .map(|p| -> VortexResult<_> {
            let f = File::create(p)?;
            Ok(PerfettoLayer::new(f).with_debug_annotations(true))
        })
        .transpose()?;

    let base = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::NONE)
        .with_ansi(false);

    if json {
        tracing_subscriber::registry()
            .with(base.json().with_filter(EnvFilter::from_default_env()))
            .with(perfetto_layer)
            .init();
    } else {
        let pretty = base
            .pretty()
            .event_format(tracing_subscriber::fmt::format().with_target(true));
        tracing_subscriber::registry()
            .with(pretty.with_filter(EnvFilter::from_default_env()))
            .with(perfetto_layer)
            .init();
    }
    Ok(())
}

// =====================================================================
// Entrypoint
// =====================================================================

fn main() -> VortexResult<()> {
    let cli = Cli::parse();
    init_tracing(cli.json, cli.perfetto.as_deref())?;

    // Build the tokio runtime: single-threaded for tokio-st, multi-threaded otherwise.
    let rt = match cli.mode {
        ScanMode::TokioSt => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| vortex::error::vortex_err!("rt build: {e}"))?,
        _ => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| vortex::error::vortex_err!("rt build: {e}"))?,
    };

    let _guard = rt.enter();

    // Session setup.
    let session = VortexSession::default().with_tokio();
    register_cuda_layout(&session);
    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?;

    let path = PathBuf::from(&cli.source);
    let file_mb = std::fs::metadata(&path)?.len() as f64 / (1024.0 * 1024.0);

    // Pre-compile CUDA kernels (PTX → SASS) so timed iterations only
    // measure I/O + decode, not one-time JIT overhead.
    eprintln!("Pre-compiling CUDA kernels...");
    session.cuda_session().preload_all_modules()?;
    eprintln!("Pre-compilation done");

    let mut cache = MmapCache::new(cuda_ctx.stream().context().clone());
    let mut times = Vec::with_capacity(cli.iterations);
    // Cached state for io_uring mode: pinned buffer + parsed footer metadata.
    let mut cached_file: Option<CachedPinnedFile> = None;

    for i in 0..cli.iterations {
        let start = Instant::now();
        let file = cache.open(&path)?;
        run_iteration(
            &rt,
            &cli.mode,
            &session,
            file,
            &mut cuda_ctx,
            &mut cached_file,
        )?;
        times.push(start.elapsed());
        tracing::info!(
            "Iteration {}/{}: {:.3}s",
            i + 1,
            cli.iterations,
            start.elapsed().as_secs_f64()
        );
    }

    print_results(&cli, file_mb, &times);
    Ok(())
}

/// Synchronous wrapper so `run_iteration` appears in frame-pointer call stacks.
/// Async fns compile to state machines that lose their name in stack unwinding.
#[inline(never)]
fn run_iteration(
    rt: &tokio::runtime::Runtime,
    mode: &ScanMode,
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
    cached_file: &mut Option<CachedPinnedFile>,
) -> VortexResult<()> {
    match mode {
        ScanMode::TokioMt | ScanMode::TokioSt => rt.block_on(tokio_scan(session, file, cuda_ctx)),
        ScanMode::Sync => sync_scan(session, file, cuda_ctx),
        ScanMode::Odirect => odirect_scan(session, file, cuda_ctx, cached_file),
        ScanMode::IoUringOdirect => io_uring_scan(session, file, cuda_ctx, cached_file, true),
        ScanMode::IoUringBuffered => io_uring_scan(session, file, cuda_ctx, cached_file, false),
        ScanMode::Fb => rt.block_on(fb_scan(session, file, cuda_ctx)),
        ScanMode::FbIoUringOdirect => fb_io_uring_scan(session, file, cuda_ctx, cached_file, true),
        ScanMode::FbIoUringBuffered => {
            fb_io_uring_scan(session, file, cuda_ctx, cached_file, false)
        }
    }
}

// =====================================================================
// Result reporting
// =====================================================================

fn print_results(cli: &Cli, file_mb: f64, times: &[Duration]) {
    let first = times[0];
    let best = times.iter().copied().min().unwrap();

    eprintln!();
    eprintln!("=== Benchmark Results ===");
    eprintln!("Source:      {}", cli.source);
    eprintln!("Iterations:  {}", cli.iterations);
    eprintln!("File size:   {file_mb:.2} MB");
    if let Some(out_mb) = cli.output_size_mb {
        eprintln!("Output size: {out_mb:.2} MB");
        eprintln!("Compression: {:.1}x", out_mb / file_mb);
    }
    eprintln!();
    print_timing("Cold (first iter)", first, file_mb, cli.output_size_mb);
    print_timing("Warm (best iter) ", best, file_mb, cli.output_size_mb);
}

fn print_timing(label: &str, duration: Duration, file_mb: f64, output_mb: Option<f64>) {
    let secs = duration.as_secs_f64();
    eprint!("{label}:  {secs:.3}s  input: {:.0} MB/s", file_mb / secs);
    if let Some(out_mb) = output_mb {
        eprint!("  output: {:.0} MB/s", out_mb / secs);
    }
    eprintln!();
}

// =====================================================================
// MmapCache + MmapFile — cached memory-mapped files with CUDA pinning
// =====================================================================

/// Cache of memory-mapped files keyed by path. Returns the same
/// [`MmapFile`] on repeated opens so CUDA page registration persists
/// across scans.
struct MmapCache {
    cuda_ctx: Arc<cudarc::driver::CudaContext>,
    entries: HashMap<PathBuf, MmapFile>,
}

impl MmapCache {
    fn new(cuda_ctx: Arc<cudarc::driver::CudaContext>) -> Self {
        Self {
            cuda_ctx,
            entries: HashMap::new(),
        }
    }

    fn open(&mut self, path: &Path) -> VortexResult<&MmapFile> {
        if !self.entries.contains_key(path) {
            self.entries.insert(
                path.to_path_buf(),
                MmapFile::open(path, self.cuda_ctx.clone())?,
            );
        }
        Ok(self.entries.get(path).unwrap())
    }
}

/// A memory-mapped file with OS prefetch hints and automatic CUDA page
/// pinning for GPU-direct NVLink-C2C access.
///
/// On open, applies `madvise(SEQUENTIAL)` and `madvise(WILLNEED)` so
/// the OS prefetches pages from disk while the scan decodes early
/// batches.
///
/// On the second call to [`buffer()`](Self::buffer), automatically pins
/// pages via `cuMemHostRegister` so subsequent GPU scans bypass ATS
/// page faults and get full NVLink-C2C bandwidth. Unregisters on drop.
struct MmapFile {
    file_path: PathBuf,
    buf: ByteBuffer,
    ptr: *const u8,
    len: usize,
    cuda_ctx: Arc<cudarc::driver::CudaContext>,
    cuda_registered: AtomicBool,
    warm: AtomicBool,
}

// SAFETY: The mmap memory is valid for the lifetime of MmapFile and
// only read (never written) after construction.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    /// Open and memory-map a file with aggressive prefetch hints.
    fn open(path: &Path, cuda_ctx: Arc<cudarc::driver::CudaContext>) -> VortexResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| vortex::error::vortex_err!("mmap failed: {e}"))?
        };
        mmap.advise(memmap2::Advice::Sequential)?;
        mmap.advise(memmap2::Advice::WillNeed)?;

        let (ptr, len) = (mmap.as_ptr(), mmap.len());
        Ok(Self {
            file_path: path.to_path_buf(),
            buf: ByteBuffer::from(Bytes::from_owner(mmap)),
            ptr,
            len,
            cuda_ctx,
            cuda_registered: AtomicBool::new(false),
            warm: AtomicBool::new(false),
        })
    }

    /// Returns the underlying buffer for use with `open_buffer()`.
    ///
    /// Returns the path to the underlying file.
    fn path(&self) -> &Path {
        &self.file_path
    }

    /// First call returns unpinned pages (cold — I/O overlaps with decode).
    /// Second call pins pages via `cuMemHostRegister` so subsequent GPU
    /// scans get full NVLink-C2C bandwidth.
    fn buffer(&self) -> VortexResult<ByteBuffer> {
        if self.warm.swap(true, Ordering::Relaxed) {
            self.ensure_cuda_registered()?;
        }
        Ok(self.buf.clone())
    }

    /// Pin pages with CUDA for GPU-direct access. No-op if already
    /// registered.
    fn ensure_cuda_registered(&self) -> VortexResult<()> {
        if self.cuda_registered.load(Ordering::Acquire) {
            return Ok(());
        }

        self.cuda_ctx
            .bind_to_thread()
            .map_err(|e| vortex::error::vortex_err!("bind CUDA context: {e}"))?;

        let result = unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                self.ptr as *mut std::ffi::c_void,
                self.len,
                cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            vortex::error::vortex_bail!("cuMemHostRegister failed: {result:?}");
        }

        self.cuda_registered.store(true, Ordering::Release);
        Ok(())
    }

    /// Unpin pages from CUDA. No-op if not registered.
    fn cuda_unregister(&self) {
        if !self.cuda_registered.load(Ordering::Acquire) {
            return;
        }
        unsafe {
            cudarc::driver::sys::cuMemHostUnregister(self.ptr as *mut std::ffi::c_void);
        }
        self.cuda_registered.store(false, Ordering::Release);
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        self.cuda_unregister();
    }
}

// =====================================================================
// Synchronous scan pipelines (sync + sync-iouring)
// =====================================================================

/// Walk the layout tree synchronously, decode each leaf via
/// `ArrayParts::decode`, build GPU dispatch plans, and launch kernels.
///
/// Shared by both [`sync_scan`] (mmap) and [`sync_iouring_scan`]
/// (io_uring).  The caller provides the [`VortexFile`] backed by
/// the appropriate buffer.
fn sync_decode_and_launch(
    vortex_file: &vortex::file::VortexFile,
    session: &VortexSession,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let footer = vortex_file.footer();
    let segment_source = vortex_file.segment_source();
    let root_layout = footer.layout();

    let mut keep_alive: Vec<BufferHandle> = Vec::new();

    for field_idx in 0..root_layout.nchildren() {
        let field_layout = root_layout.child(field_idx)?;
        let ptype = match field_layout.dtype() {
            DType::Primitive(p, _) => *p,
            _ => continue,
        };

        let mut leaves = Vec::new();
        collect_flat_leaves(&field_layout, &mut leaves)?;
        if leaves.is_empty() {
            continue;
        }

        let mut plans: Vec<DynamicDispatchPlan> = Vec::with_capacity(leaves.len());
        let mut chunk_lens: Vec<usize> = Vec::with_capacity(leaves.len());

        for leaf in &leaves {
            let (array_tree, segment) = resolve_leaf_segment(leaf, segment_source.as_ref())?;
            let parts = if array_tree.is_empty() {
                ArrayParts::try_from(segment)?
            } else {
                ArrayParts::from_flatbuffer_and_segment(array_tree, segment)?
            };
            let array = parts.decode(&leaf.dtype, leaf.row_count as usize, &leaf.ctx, session)?;

            match dynamic_dispatch::build_plan(&array, cuda_ctx) {
                Ok((plan, buffers)) => {
                    plans.push(plan);
                    chunk_lens.push(leaf.row_count as usize);
                    keep_alive.extend(buffers);
                }
                Err(e) => {
                    tracing::debug!(error = %e, "plan build failed, skipping");
                }
            }
        }

        if plans.is_empty() {
            continue;
        }

        launch_column_kernel(&plans, &chunk_lens, ptype, cuda_ctx)?;
    }

    cuda_ctx.synchronize_stream()?;
    drop(keep_alive);
    Ok(())
}

/// Allocate GPU output, upload plans, and launch the dynamic dispatch
/// kernel for one column's worth of chunks.
fn launch_column_kernel(
    plans: &[DynamicDispatchPlan],
    chunk_lens: &[usize],
    ptype: PType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let kernel_ptype = unsigned_ptype(ptype);
    let smem_bytes = smem_for_ptype(&plans[0], ptype);

    let total_elems: usize = chunk_lens.iter().map(|l| l.next_multiple_of(1024)).sum();
    let output = CudaDeviceBuffer::new(ctx.device_alloc::<u32>(total_elems)?);
    let output_view = output.as_view::<u32>();
    let (base_ptr, _out_guard) = output_view.device_ptr(ctx.stream());

    let mut plans = plans.to_vec();
    let mut offset: u64 = 0;
    for (plan, &len) in plans.iter_mut().zip(chunk_lens) {
        plan.output_ptr = base_ptr + offset * 4;
        plan.array_len = len as u64;
        offset += len.next_multiple_of(1024) as u64;
    }

    let device_plans = ctx
        .stream()
        .clone_htod(&plans)
        .map_err(|e| vortex::error::vortex_err!("copy plans to device: {e}"))?;
    let (plans_ptr, _plans_guard) = device_plans.device_ptr(ctx.stream());

    let ptype_str = kernel_ptype.to_string();
    let function: CudaFunction =
        ctx.load_function_with_suffixes("dynamic_dispatch", &["multi", &ptype_str])?;

    #[allow(clippy::cast_possible_truncation)]
    let max_blocks = chunk_lens
        .iter()
        .map(|l| l.div_ceil(2048) as u32)
        .max()
        .unwrap_or(0);
    #[allow(clippy::cast_possible_truncation)]
    let num_chunks = plans.len() as u32;

    let mut builder = ctx.stream().launch_builder(&function);
    builder.arg(&plans_ptr);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (max_blocks, num_chunks, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: smem_bytes,
            })
            .map_err(|e| vortex::error::vortex_err!("kernel launch failed: {e}"))?;
    }

    drop((_out_guard, _plans_guard));
    Ok(())
}

// ── sync (mmap) ─────────────────────────────────────────────────────

/// Synchronous scan backed by mmap.  Pins pages via `cuMemHostRegister`
/// for GPU zero-copy access, then calls [`sync_decode_and_launch`].
fn sync_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    file.ensure_cuda_registered()?;
    let vortex_file = session.open_options().open_buffer(file.buffer()?)?;
    sync_decode_and_launch(&vortex_file, session, cuda_ctx)
}

/// Synchronous scan with `pread` + `O_DIRECT` (no io_uring).
/// Reads the file sequentially into a CUDA-pinned buffer on the first
/// call, reuses the cached buffer on subsequent calls.
fn odirect_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
    state: &mut Option<CachedPinnedFile>,
) -> VortexResult<()> {
    if state.is_none() {
        *state = Some(CachedPinnedFile {
            pinned_bytes: pread_odirect_file(file.path())?,
        });
    }
    let vortex_file = open_cached_file(session, state)?;
    sync_decode_and_launch(&vortex_file, session, cuda_ctx)
}

// ── sync-iouring ────────────────────────────────────────────────────

/// CUDA-pinned buffer allocated via `cuMemAllocHost`. Freed via
/// `cuMemFreeHost` on drop.
struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl AsRef<[u8]> for PinnedBuf {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// Cached CUDA-pinned buffer, persists across iterations.
/// Used by all modes that read into a pinned buffer (O_DIRECT, io_uring).
struct CachedPinnedFile {
    pinned_bytes: ByteBuffer,
}

/// Allocate a 4096-aligned, CUDA-pinned buffer of `size` bytes.
fn alloc_cuda_pinned(size: usize) -> VortexResult<*mut u8> {
    let mut dev_ptr: cudarc::driver::sys::CUdeviceptr = 0;
    unsafe {
        let r = cudarc::driver::sys::cuMemAllocHost_v2(
            &mut dev_ptr as *mut _ as *mut *mut std::ffi::c_void,
            size,
        );
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            vortex::error::vortex_bail!("cuMemAllocHost failed: {r:?}");
        }
    }
    Ok(dev_ptr as *mut u8)
}

/// Read an entire file into a CUDA-pinned buffer via sequential `pread`
/// with `O_DIRECT`.  No io_uring — just a plain read loop.
fn pread_odirect_file(path: &Path) -> VortexResult<ByteBuffer> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;

    let file_len = std::fs::metadata(path)?.len() as usize;
    let buf_size = (file_len + 4095) & !4095;
    let ptr = alloc_cuda_pinned(buf_size)?;

    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?;
    let fd = f.as_raw_fd();

    let mut offset: usize = 0;
    while offset < buf_size {
        let chunk = (buf_size - offset).min(1024 * 1024);
        let n = unsafe {
            libc::pread(
                fd,
                ptr.add(offset) as *mut std::ffi::c_void,
                chunk,
                offset as libc::off_t,
            )
        };
        if n < 0 {
            drop(f);
            vortex::error::vortex_bail!("pread failed: {}", std::io::Error::last_os_error());
        }
        if n == 0 {
            break;
        }
        offset += n as usize;
    }
    drop(f);

    Ok(ByteBuffer::from(Bytes::from_owner(PinnedBuf {
        ptr,
        len: file_len,
    })))
}

/// Read an entire file into a CUDA-pinned buffer via `io_uring`.
///
/// When `o_direct` is true, opens with `O_DIRECT` to bypass the page cache.
/// When false, uses buffered I/O (page cache) for warm-friendly reads.
///
/// Splits the file into 1 MB chunks submitted with queue depth 128 and
/// pre-registered buffers/fds to saturate NVMe parallelism.
fn io_uring_read_file(path: &Path, o_direct: bool) -> VortexResult<ByteBuffer> {
    use io_uring::{IoUring, opcode, types};
    use std::os::unix::io::AsRawFd;

    let t0 = Instant::now();

    let file_len = std::fs::metadata(path)?.len() as usize;
    let buf_size = (file_len + 4095) & !4095;

    let ptr = alloc_cuda_pinned(buf_size)?;
    let t_alloc_pin = t0.elapsed();

    let fd = {
        use std::os::unix::fs::OpenOptionsExt;
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true);
        if o_direct {
            opts.custom_flags(libc::O_DIRECT);
        }
        let f = opts.open(path)?;
        let raw = f.as_raw_fd();
        std::mem::forget(f);
        raw
    };

    const CHUNK_SIZE: usize = 1024 * 1024; // 1 MB per request
    const QUEUE_DEPTH: u32 = 128;

    // Build io_uring with optimizations:
    // - single_issuer: only one thread submits, enables kernel fast path
    let mut ring: IoUring = IoUring::builder()
        .setup_single_issuer()
        .build(QUEUE_DEPTH)
        .map_err(|e| vortex::error::vortex_err!("io_uring_setup: {e}"))?;

    // Register the file descriptor so the kernel skips fd table lookups.
    ring.submitter()
        .register_files(&[fd])
        .map_err(|e| vortex::error::vortex_err!("register_files: {e}"))?;
    let fixed_fd = types::Fixed(0);

    // Register the pinned buffer so the kernel can skip per-request
    // page pinning — the pages are already pinned by CUDA.
    let iov = libc::iovec {
        iov_base: ptr as *mut std::ffi::c_void,
        iov_len: buf_size,
    };
    unsafe {
        ring.submitter()
            .register_buffers(&[iov])
            .map_err(|e| vortex::error::vortex_err!("register_buffers: {e}"))?;
    }
    let t_setup = t0.elapsed();

    let num_chunks = buf_size.div_ceil(CHUNK_SIZE);
    let mut submitted: usize = 0;
    let mut completed: usize = 0;

    // Submit chunks using ReadFixed (pre-registered buffer) and Fixed fd.
    while completed < num_chunks {
        // Fill the submission queue (scoped so the mutable borrow ends).
        {
            let mut sq = ring.submission();
            while submitted < num_chunks {
                let chunk_offset = submitted * CHUNK_SIZE;
                let chunk_len = CHUNK_SIZE.min(buf_size - chunk_offset);

                let entry = opcode::ReadFixed::new(
                    fixed_fd,
                    unsafe { ptr.add(chunk_offset) },
                    chunk_len as u32,
                    0, // buf_index — we registered one buffer at index 0
                )
                .offset(chunk_offset as u64)
                .build()
                .user_data(submitted as u64);

                match unsafe { sq.push(&entry) } {
                    Ok(()) => submitted += 1,
                    Err(_) => break,
                }
            }
        } // sq dropped here, mutable borrow released

        let want = if submitted == num_chunks {
            submitted - completed
        } else {
            1
        };
        ring.submit_and_wait(want)
            .map_err(|e| vortex::error::vortex_err!("io_uring wait: {e}"))?;

        let cq = ring.completion();
        for cqe in cq {
            if cqe.result() < 0 {
                unsafe { libc::close(fd) };
                vortex::error::vortex_bail!(
                    "io_uring read chunk {}: {}",
                    cqe.user_data(),
                    std::io::Error::from_raw_os_error(-cqe.result())
                );
            }
            completed += 1;
        }
    }

    let t_io = t0.elapsed();

    // Cleanup io_uring registrations.
    ring.submitter().unregister_files().ok();
    ring.submitter().unregister_buffers().ok();
    unsafe { libc::close(fd) };

    eprintln!(
        "  [io_uring_read_file] alloc+pin={:.3}ms  setup={:.3}ms  io={:.3}ms  total={:.3}ms  \
         file={:.1}MB  io_throughput={:.0} MB/s",
        t_alloc_pin.as_secs_f64() * 1e3,
        (t_setup - t_alloc_pin).as_secs_f64() * 1e3,
        (t_io - t_setup).as_secs_f64() * 1e3,
        t_io.as_secs_f64() * 1e3,
        file_len as f64 / 1e6,
        file_len as f64 / (t_io - t_setup).as_secs_f64() / 1e6,
    );

    Ok(ByteBuffer::from(Bytes::from_owner(PinnedBuf {
        ptr,
        len: file_len,
    })))
}

/// Ensure the cached pinned buffer is populated.  On the first call,
/// reads the file via `io_uring` into a `cuMemAllocHost` buffer.
/// Subsequent calls return the cached buffer immediately.
fn ensure_cached_file(
    file: &MmapFile,
    state: &mut Option<CachedPinnedFile>,
    o_direct: bool,
) -> VortexResult<()> {
    if state.is_none() {
        *state = Some(CachedPinnedFile {
            pinned_bytes: io_uring_read_file(file.path(), o_direct)?,
        });
    }
    Ok(())
}

/// Open a [`VortexFile`] from the cached pinned buffer.
fn open_cached_file(
    session: &VortexSession,
    state: &Option<CachedPinnedFile>,
) -> VortexResult<vortex::file::VortexFile> {
    session
        .open_options()
        .open_buffer(state.as_ref().unwrap().pinned_bytes.clone())
}

/// Synchronous scan with `io_uring` + `O_DIRECT` for cold reads.
fn io_uring_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
    state: &mut Option<CachedPinnedFile>,
    o_direct: bool,
) -> VortexResult<()> {
    ensure_cached_file(file, state, o_direct)?;
    let vortex_file = open_cached_file(session, state)?;
    sync_decode_and_launch(&vortex_file, session, cuda_ctx)
}

/// Run a single scan iteration: open the buffer, decode all batches
/// through the GPU pipeline, and synchronize.
async fn tokio_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let vortex_file = session.open_options().open_buffer(file.buffer()?)?;

    let mut batches = vortex_file
        .scan()?
        .with_split_by(SplitBy::RowCount(2_500_000))
        .into_array_stream()?;

    // Collect decoded batches to keep mmap-backed buffers alive until
    // the GPU stream is synchronized.
    let mut keep_alive = Vec::new();
    while let Some(batch) = batches.next().await.transpose()? {
        decode_batch(&batch, cuda_ctx)?;
        keep_alive.push(batch);
    }

    cuda_ctx.synchronize_stream()?;
    drop(keep_alive);
    Ok(())
}

/// Decode a single struct batch: build dynamic-dispatch plans for each
/// primitive column and launch fused GPU decompression kernels.
///
/// Columns are processed sequentially to pipeline CPU plan-building
/// with GPU execution (GPU works on column N while CPU builds N+1).
fn decode_batch(batch: &vortex::array::ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<()> {
    let struct_arr = batch
        .as_opt::<StructVTable>()
        .ok_or_else(|| vortex::error::vortex_err!("expected struct batch"))?;

    for field in &struct_arr.clone().into_fields() {
        let ptype = match field.dtype() {
            DType::Primitive(p, _) => *p,
            _ => continue,
        };

        if field.encoding_id() == ChunkedVTable::ID {
            let chunked = field
                .clone()
                .try_into::<ChunkedVTable>()
                .map_err(|e| vortex::error::vortex_err!("ChunkedVTable cast: {e}"))?;
            decode_column(&chunked.chunks().to_vec(), ptype, ctx)?;
        } else {
            decode_column(std::slice::from_ref(field), ptype, ctx)?;
        }
    }
    Ok(())
}

/// Build plans for all chunks of a column and dispatch them in a single
/// `dynamic_dispatch_multi` kernel launch with a 2D grid (blockIdx.y =
/// chunk, blockIdx.x = block within chunk).
fn decode_column(
    chunks: &[vortex::array::ArrayRef],
    ptype: PType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    if chunks.is_empty() {
        return Ok(());
    }

    // --- Phase 1: build plans on host --------------------------------
    let mut plans: Vec<DynamicDispatchPlan> = Vec::with_capacity(chunks.len());
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(chunks.len());
    let mut keep_alive: Vec<BufferHandle> = Vec::new();

    for chunk in chunks {
        match dynamic_dispatch::build_plan(chunk, ctx) {
            Ok((plan, buffers)) => {
                plans.push(plan);
                chunk_lens.push(chunk.len());
                keep_alive.extend(buffers);
            }
            Err(e) => {
                tracing::debug!(error = %e, "plan build failed, skipping chunk");
            }
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    let kernel_ptype = unsigned_ptype(ptype);
    let smem_bytes = smem_for_ptype(&plans[0], ptype);

    // --- Phase 2: allocate output buffer for the entire column --------
    let total_elems: usize = chunk_lens
        .iter()
        .map(|len| len.next_multiple_of(1024))
        .sum();
    let output = CudaDeviceBuffer::new(ctx.device_alloc::<u32>(total_elems)?);
    let output_view = output.as_view::<u32>();
    let (base_ptr, _output_guard) = output_view.device_ptr(ctx.stream());

    // --- Phase 3: embed output pointers into plans --------------------
    let mut offset: u64 = 0;
    for (plan, &len) in plans.iter_mut().zip(&chunk_lens) {
        plan.output_ptr = base_ptr + offset * 4;
        plan.array_len = len as u64;
        offset += len.next_multiple_of(1024) as u64;
    }

    // --- Phase 4: upload plans to device ------------------------------
    let device_plans = ctx
        .stream()
        .clone_htod(&plans)
        .map_err(|e| vortex::error::vortex_err!("copy plans to device: {e}"))?;
    let (plans_ptr, _plans_guard) = device_plans.device_ptr(ctx.stream());

    // --- Phase 5: launch kernel --------------------------------------
    let ptype_str = kernel_ptype.to_string();
    let function: CudaFunction =
        ctx.load_function_with_suffixes("dynamic_dispatch", &["multi", &ptype_str])?;

    #[allow(clippy::cast_possible_truncation)]
    let max_blocks = chunk_lens
        .iter()
        .map(|len| len.div_ceil(2048) as u32)
        .max()
        .unwrap_or(0);

    #[allow(clippy::cast_possible_truncation)]
    let num_chunks = plans.len() as u32;

    let mut builder = ctx.stream().launch_builder(&function);
    builder.arg(&plans_ptr);

    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (max_blocks, num_chunks, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: smem_bytes,
            })
            .map_err(|e| vortex::error::vortex_err!("kernel launch failed: {e}"))?;
    }

    drop((_output_guard, _plans_guard, keep_alive));
    Ok(())
}

// =====================================================================
// Direct-from-flatbuffer scan (zero deserialization)
// =====================================================================

/// Scan a [`MmapFile`] by walking the layout tree and building GPU dispatch
/// plans directly from flatbuffer metadata — no ArrayRef construction.
/// Direct-fb scan with io_uring + O_DIRECT for cold reads.
fn fb_io_uring_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
    state: &mut Option<CachedPinnedFile>,
    o_direct: bool,
) -> VortexResult<()> {
    ensure_cached_file(file, state, o_direct)?;
    let vortex_file = open_cached_file(session, state)?;
    fb_decode_and_launch(&vortex_file, cuda_ctx)
}

/// Walk the layout tree, build GPU plans directly from flatbuffer metadata
/// (no ArrayRef construction), and launch kernels.
fn fb_decode_and_launch(
    vortex_file: &vortex::file::VortexFile,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    let footer = vortex_file.footer();
    let segment_source = vortex_file.segment_source();
    let root_layout = footer.layout();

    let mut keep_alive: Vec<BufferHandle> = Vec::new();
    for field_idx in 0..root_layout.nchildren() {
        let field_layout = root_layout.child(field_idx)?;
        let ptype = match field_layout.dtype() {
            DType::Primitive(p, _) => *p,
            _ => continue,
        };
        let mut leaves = Vec::new();
        collect_flat_leaves(&field_layout, &mut leaves)?;
        if !leaves.is_empty() {
            decode_column_direct(
                &leaves,
                ptype,
                segment_source.as_ref(),
                cuda_ctx,
                &mut keep_alive,
            )?;
        }
    }

    cuda_ctx.synchronize_stream()?;
    drop(keep_alive);
    Ok(())
}

async fn fb_scan(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    file.ensure_cuda_registered()?;
    let vortex_file = session.open_options().open_buffer(file.buffer()?)?;
    fb_decode_and_launch(&vortex_file, cuda_ctx)
}

/// Leaf metadata extracted from a `CudaFlatLayout` or `FlatLayout`.
struct FlatLeaf {
    array_tree: ByteBuffer,
    segment_id: vortex::layout::segments::SegmentId,
    host_buffers: Arc<VortexHashMap<u32, ByteBuffer>>,
    row_count: u64,
    dtype: DType,
    ctx: ArrayContext,
}

/// Recursively collect flat leaves from the layout tree.
fn collect_flat_leaves(layout: &LayoutRef, out: &mut Vec<FlatLeaf>) -> VortexResult<()> {
    if let Some(cuda_flat) = layout.as_opt::<CudaFlatVTable>() {
        out.push(FlatLeaf {
            array_tree: cuda_flat.array_tree().clone(),
            segment_id: cuda_flat.segment_id(),
            host_buffers: cuda_flat.host_buffers().clone(),
            row_count: cuda_flat.row_count(),
            dtype: cuda_flat.dtype().clone(),
            ctx: cuda_flat.array_ctx().clone(),
        });
        return Ok(());
    }
    if let Some(flat) = layout.as_opt::<FlatVTable>() {
        out.push(FlatLeaf {
            array_tree: flat.array_tree().cloned().unwrap_or_else(ByteBuffer::empty),
            segment_id: flat.segment_id(),
            host_buffers: Arc::new(VortexHashMap::new()),
            row_count: flat.row_count(),
            dtype: flat.dtype().clone(),
            ctx: flat.array_ctx().clone(),
        });
        return Ok(());
    }
    for i in 0..layout.nchildren() {
        if matches!(layout.child_type(i), LayoutChildType::Auxiliary(_)) {
            continue;
        }
        collect_flat_leaves(&layout.child(i)?, out)?;
    }
    Ok(())
}

/// Resolve a leaf's segment into (array_tree, segment_data).
fn resolve_leaf_segment(
    leaf: &FlatLeaf,
    segment_source: &dyn SegmentSource,
) -> VortexResult<(ByteBuffer, BufferHandle)> {
    let segment = futures::executor::block_on(segment_source.request(leaf.segment_id))?;
    let segment = segment.ensure_aligned(Alignment::none())?;

    if leaf.array_tree.is_empty() {
        let host_buf = segment.try_to_host_sync()?;
        if host_buf.len() < 4 {
            vortex::error::vortex_bail!("segment too short for array tree suffix");
        }
        #[allow(clippy::cast_possible_truncation)]
        let fb_len = u32::from_le_bytes(
            host_buf.as_slice()[host_buf.len() - 4..]
                .try_into()
                .map_err(|_| vortex::error::vortex_err!("failed to read flatbuffer length"))?,
        ) as usize;
        let fb_off = host_buf.len() - 4 - fb_len;
        Ok((
            host_buf.slice(fb_off..fb_off + fb_len),
            BufferHandle::new_host(host_buf.slice(0..fb_off).aligned(Alignment::none())),
        ))
    } else {
        Ok((leaf.array_tree.clone(), segment))
    }
}

/// Build plans directly from flatbuffer metadata and launch the GPU kernel.
#[allow(clippy::cast_possible_truncation)]
fn decode_column_direct(
    leaves: &[FlatLeaf],
    ptype: PType,
    segment_source: &dyn SegmentSource,
    ctx: &mut CudaExecutionCtx,
    keep_alive: &mut Vec<BufferHandle>,
) -> VortexResult<()> {
    if leaves.is_empty() {
        return Ok(());
    }

    let dtype = leaves[0].dtype.clone();
    let mut plans: Vec<DynamicDispatchPlan> = Vec::with_capacity(leaves.len());
    let mut chunk_lens: Vec<usize> = Vec::with_capacity(leaves.len());

    for leaf in leaves {
        let (array_tree, segment) = resolve_leaf_segment(leaf, segment_source)?;
        match build_plan_from_flatbuffer(
            &array_tree,
            &segment,
            &leaf.host_buffers,
            leaf.row_count,
            &dtype,
            &leaf.ctx,
            ctx,
        ) {
            Ok((plan, bufs)) => {
                plans.push(plan);
                chunk_lens.push(leaf.row_count as usize);
                keep_alive.extend(bufs);
            }
            Err(e) => {
                tracing::debug!(error = %e, "direct-fb plan build failed, skipping");
            }
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    let kernel_ptype = unsigned_ptype(ptype);
    let smem_bytes = smem_for_ptype(&plans[0], ptype);

    let total_elems: usize = chunk_lens.iter().map(|l| l.next_multiple_of(1024)).sum();
    let output = CudaDeviceBuffer::new(ctx.device_alloc::<u32>(total_elems)?);
    let output_view = output.as_view::<u32>();
    let (base_ptr, _out_guard) = output_view.device_ptr(ctx.stream());

    let mut offset: u64 = 0;
    for (plan, &len) in plans.iter_mut().zip(&chunk_lens) {
        plan.output_ptr = base_ptr + offset * 4;
        plan.array_len = len as u64;
        offset += len.next_multiple_of(1024) as u64;
    }

    let device_plans = ctx
        .stream()
        .clone_htod(&plans)
        .map_err(|e| vortex::error::vortex_err!("copy plans to device: {e}"))?;
    let (plans_ptr, _plans_guard) = device_plans.device_ptr(ctx.stream());

    let ptype_str = kernel_ptype.to_string();
    let function: CudaFunction =
        ctx.load_function_with_suffixes("dynamic_dispatch", &["multi", &ptype_str])?;

    let max_blocks = chunk_lens
        .iter()
        .map(|l| l.div_ceil(2048) as u32)
        .max()
        .unwrap_or(0);
    let num_chunks = plans.len() as u32;

    let mut builder = ctx.stream().launch_builder(&function);
    builder.arg(&plans_ptr);
    unsafe {
        builder
            .launch(LaunchConfig {
                grid_dim: (max_blocks, num_chunks, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: smem_bytes,
            })
            .map_err(|e| vortex::error::vortex_err!("kernel launch failed: {e}"))?;
    }

    drop((_out_guard, _plans_guard));
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================

fn smem_for_ptype(plan: &DynamicDispatchPlan, ptype: PType) -> u32 {
    match ptype.byte_width() {
        1 => plan.shared_mem_bytes::<u8>(),
        2 => plan.shared_mem_bytes::<u16>(),
        8 => plan.shared_mem_bytes::<u64>(),
        _ => plan.shared_mem_bytes::<u32>(),
    }
}

/// Map any PType to the unsigned integer of the same byte width.
/// The `dynamic_dispatch` kernel is only instantiated for unsigned types.
fn unsigned_ptype(ptype: PType) -> PType {
    // to_unsigned() handles signed ints; floats need byte-width mapping.
    match ptype {
        PType::F16 | PType::F32 | PType::F64 => match ptype.byte_width() {
            2 => PType::U16,
            8 => PType::U64,
            _ => PType::U32,
        },
        other => other.to_unsigned(),
    }
}

/// Synchronously decode all column leaves from a [`VortexFile`] and
/// assemble them into struct batch `ArrayRef`s — matching the output
/// shape of the tokio-mt scan pipeline.
#[cfg(test)]
fn sync_decode_to_struct_batches(
    vortex_file: &vortex::file::VortexFile,
    session: &VortexSession,
) -> VortexResult<Vec<vortex::array::ArrayRef>> {
    use vortex::array::IntoArray;
    use vortex::array::arrays::ChunkedArray;

    let footer = vortex_file.footer();
    let segment_source = vortex_file.segment_source();
    let root_layout = footer.layout();
    let dtype = footer.dtype().clone();
    let struct_fields = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex::error::vortex_err!("root dtype is not a struct"))?;
    let field_names: Vec<String> = struct_fields
        .names()
        .iter()
        .map(|n| n.to_string())
        .collect();

    // Decode each column's leaves into a single ChunkedArray.
    let mut column_arrays: Vec<(&str, vortex::array::ArrayRef)> = Vec::new();
    let mut col_name_idx = 0;
    for field_idx in 0..root_layout.nchildren() {
        let field_layout = root_layout.child(field_idx)?;

        let mut leaves = Vec::new();
        collect_flat_leaves(&field_layout, &mut leaves)?;
        if leaves.is_empty() {
            continue;
        }

        let mut chunks = Vec::new();
        for leaf in &leaves {
            let (array_tree, segment) = resolve_leaf_segment(leaf, segment_source.as_ref())?;
            let parts = if array_tree.is_empty() {
                ArrayParts::try_from(segment)?
            } else {
                ArrayParts::from_flatbuffer_and_segment(array_tree, segment)?
            };
            let array = parts.decode(&leaf.dtype, leaf.row_count as usize, &leaf.ctx, session)?;
            chunks.push(array);
        }

        let col_array = if chunks.len() == 1 {
            chunks.into_iter().next().unwrap()
        } else {
            ChunkedArray::from_iter(chunks).into_array()
        };

        // Leak is fine for tests — the &str lives for the process lifetime.
        let name_str: &'static str = Box::leak(field_names[col_name_idx].clone().into_boxed_str());
        column_arrays.push((name_str, col_array));
        col_name_idx += 1;
    }

    let batch = vortex::array::arrays::StructArray::from_fields(&column_arrays)?.into_array();
    Ok(vec![batch])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use vortex::VortexSessionDefault;
    use vortex::array::arrays::{PrimitiveArray, StructArray};
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::array::{ArrayRef, IntoArray};
    use vortex::buffer::Buffer;
    use vortex::dtype::Nullability::NonNullable;
    use vortex::error::VortexResult;
    use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt, WriteStrategyBuilder};
    use vortex::io::session::RuntimeSessionExt;
    use vortex::scan::SplitBy;
    use vortex::session::VortexSession;
    use vortex_cuda::layout::{CudaFlatLayoutStrategy, register_cuda_layout};

    use super::*;

    /// Shared session + CUDA context for tests.
    fn test_session() -> (VortexSession, CudaExecutionCtx) {
        let session = VortexSession::default().with_tokio();
        register_cuda_layout(&session);
        let cuda_ctx = CudaSession::create_execution_ctx(&session).unwrap();
        (session, cuda_ctx)
    }

    /// Write a small CUDA-compatible vortex file with two primitive columns.
    fn write_test_file(path: &Path, session: &VortexSession) -> VortexResult<()> {
        let batch = StructArray::from_fields(&[
            (
                "a",
                PrimitiveArray::new(
                    Buffer::from(vec![1i32, 2, 3, 4, 5, 6, 7, 8]),
                    NonNullable.into(),
                )
                .into_array(),
            ),
            (
                "b",
                PrimitiveArray::new(
                    Buffer::from(vec![10i64, 20, 30, 40, 50, 60, 70, 80]),
                    NonNullable.into(),
                )
                .into_array(),
            ),
        ])?
        .into_array();

        let strategy = WriteStrategyBuilder::default()
            .with_cuda_compatible_encodings()
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()))
            .build();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mut out = tokio::fs::File::create(path).await?;
            session
                .write_options()
                .with_strategy(strategy)
                .write(
                    &mut out,
                    ArrayStreamAdapter::new(
                        batch.dtype().clone(),
                        futures::stream::once(async { Ok(batch) }),
                    ),
                )
                .await
        })?;
        Ok(())
    }

    /// Decode via the standard tokio multi-threaded async scan pipeline.
    fn tokio_mt_decode(path: &Path, session: &VortexSession) -> VortexResult<Vec<ArrayRef>> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let buf = ByteBuffer::from(Bytes::from(std::fs::read(path)?));
            let vf = session.open_options().open_buffer(buf)?;
            let mut stream = vf
                .scan()?
                .with_split_by(SplitBy::RowCount(1_000_000))
                .into_array_stream()?;
            let mut out = Vec::new();
            while let Some(batch) = stream.next().await.transpose()? {
                out.push(batch);
            }
            Ok(out)
        })
    }

    /// Decode via the synchronous io_uring path.
    fn sync_io_uring_decode(path: &Path, session: &VortexSession) -> VortexResult<Vec<ArrayRef>> {
        let buf = io_uring_read_file(path, true)?;
        let vf = session.open_options().open_buffer(buf)?;
        sync_decode_to_struct_batches(&vf, session)
    }

    #[test]
    fn test_io_uring_read_file_matches_std_read() -> VortexResult<()> {
        let dir = std::env::temp_dir().join("gpu_scan_bench_test_io_uring");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("io_uring_test.bin");

        // Write a file with known content.
        let n = 256 * 1024; // 256 KB — large enough to exercise chunked io_uring reads
        let expected: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &expected)?;

        // Need CUDA context for cuMemAllocHost.
        let (_session, _cuda_ctx) = test_session();

        // Read via io_uring + O_DIRECT.
        let buf = io_uring_read_file(&path, true)?;

        assert_eq!(buf.len(), expected.len(), "length mismatch");
        assert_eq!(buf.as_slice(), expected.as_slice(), "content mismatch");

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
        Ok(())
    }

    #[test]
    fn test_tokio_mt_vs_sync_io_uring_produce_same_data() -> VortexResult<()> {
        let dir = std::env::temp_dir().join("gpu_scan_bench_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.vortex");

        let (session, _cuda_ctx) = test_session();
        write_test_file(&path, &session)?;

        let tokio_batches = tokio_mt_decode(&path, &session)?;
        let uring_batches = sync_io_uring_decode(&path, &session)?;

        assert_eq!(tokio_batches.len(), 1, "expected 1 tokio-mt batch");
        assert_eq!(uring_batches.len(), 1, "expected 1 sync-iouring batch");
        vortex_array::assert_arrays_eq!(tokio_batches[0], uring_batches[0]);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
        Ok(())
    }
}
