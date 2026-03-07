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
    /// Standard path: open_buffer → scan → decode batches → GPU kernels.
    Scan,
    /// Same as Scan but forced onto a single-threaded tokio runtime.
    /// Isolates Arc contention cost vs inherent Arc cost.
    ScanSt,
    /// Zero-deserialization: walk layout tree → build plans directly from
    /// flatbuffer metadata → GPU kernels. Bypasses ArrayRef construction.
    DirectFb,
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

    /// Scan mode: 'scan' (default) or 'direct-fb' (zero-deserialization).
    #[arg(long, value_enum, default_value_t = ScanMode::Scan)]
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

    // Build the tokio runtime: single-threaded for scan-st, multi-threaded otherwise.
    let rt = match cli.mode {
        ScanMode::ScanSt => tokio::runtime::Builder::new_current_thread()
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

    for i in 0..cli.iterations {
        let start = Instant::now();
        let file = cache.open(&path)?;
        run_iteration(&rt, &cli.mode, &session, file, &mut cuda_ctx)?;
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
) -> VortexResult<()> {
    match mode {
        ScanMode::Scan | ScanMode::ScanSt => rt.block_on(scan_file(session, file, cuda_ctx)),
        ScanMode::DirectFb => rt.block_on(direct_scan_file(session, file, cuda_ctx)),
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
// Scan pipeline
// =====================================================================

/// Run a single scan iteration: open the buffer, decode all batches
/// through the GPU pipeline, and synchronize.
async fn scan_file(
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
async fn direct_scan_file(
    session: &VortexSession,
    file: &MmapFile,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<()> {
    // Direct-fb embeds raw host pointers into GPU plans — pages must be
    // pinned via cuMemHostRegister (ATS faults are not sufficient here).
    file.ensure_cuda_registered()?;
    let vortex_file = session.open_options().open_buffer(file.buffer()?)?;
    let footer = vortex_file.footer();
    let segment_source = vortex_file.segment_source();
    let root_layout = footer.layout();

    let mut columns: Vec<(PType, Vec<FlatLeaf>)> = Vec::new();
    for field_idx in 0..root_layout.nchildren() {
        let field_layout = root_layout.child(field_idx)?;
        let ptype = match field_layout.dtype() {
            DType::Primitive(p, _) => *p,
            _ => continue,
        };
        let mut leaves = Vec::new();
        collect_flat_leaves(&field_layout, &mut leaves)?;
        if !leaves.is_empty() {
            columns.push((ptype, leaves));
        }
    }

    let mut keep_alive: Vec<BufferHandle> = Vec::new();
    for (ptype, leaves) in &columns {
        decode_column_direct(
            leaves,
            *ptype,
            segment_source.as_ref(),
            cuda_ctx,
            &mut keep_alive,
        )?;
    }

    cuda_ctx.synchronize_stream()?;
    drop(keep_alive);
    Ok(())
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
