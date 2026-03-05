// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[cuda_not_available]
fn main() {}

#[cuda_available]
fn main() -> vortex::error::VortexResult<()> {
    cuda_main::main()
}

#[cuda_available]
mod cuda_main {
    use std::fs::File;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use bytes::Bytes;
    use clap::Parser;
    use cudarc::driver::CudaFunction;
    use cudarc::driver::DevicePtr;
    use cudarc::driver::LaunchConfig;
    use cudarc::driver::PushKernelArg;
    use futures::StreamExt;
    use tracing_perfetto::PerfettoLayer;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::Layer;
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use vortex::VortexSessionDefault;
    use vortex::array::arrays::ChunkedVTable;
    use vortex::array::arrays::StructVTable;
    use vortex::array::buffer::BufferHandle;
    use vortex::buffer::ByteBuffer;
    use vortex::dtype::DType;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::scan::SplitBy;
    use vortex::session::VortexSession;
    use vortex_cuda::CudaDeviceBuffer;
    use vortex_cuda::CudaExecutionCtx;
    use vortex_cuda::CudaSession;
    use vortex_cuda::dynamic_dispatch;
    use vortex_cuda::dynamic_dispatch::DynamicDispatchPlan;
    use vortex_cuda::layout::register_cuda_layout;

    #[derive(Parser)]
    #[command(
        name = "gpu-scan-bench",
        about = "Benchmark GPU scans of CUDA-compatible Vortex files"
    )]
    struct Cli {
        /// Local path to a .vortex file.
        source: String,

        /// Number of timed scan iterations (excludes the warmup iteration).
        #[arg(long, default_value_t = 1)]
        iterations: usize,

        /// Path to write Perfetto trace output. If omitted, no trace file is written.
        #[arg(long)]
        perfetto: Option<PathBuf>,

        /// Output logs as JSON.
        #[arg(long)]
        json: bool,
    }

    fn init_tracing(json: bool, perfetto_path: Option<&Path>) -> VortexResult<()> {
        let perfetto_layer = if let Some(perfetto_path) = perfetto_path {
            let perfetto_file = File::create(perfetto_path)?;
            Some(PerfettoLayer::new(perfetto_file).with_debug_annotations(true))
        } else {
            None
        };

        let base_log_layer = tracing_subscriber::fmt::layer()
            .with_span_events(FmtSpan::NONE)
            .with_ansi(false);

        if json {
            let log_layer = base_log_layer.json();
            tracing_subscriber::registry()
                .with(log_layer.with_filter(EnvFilter::from_default_env()))
                .with(perfetto_layer)
                .init();
        } else {
            let log_layer = base_log_layer
                .pretty()
                .event_format(tracing_subscriber::fmt::format().with_target(true));
            tracing_subscriber::registry()
                .with(log_layer.with_filter(EnvFilter::from_default_env()))
                .with(perfetto_layer)
                .init();
        }

        Ok(())
    }

    #[tokio::main]
    pub async fn main() -> VortexResult<()> {
        let cli = Cli::parse();
        init_tracing(cli.json, cli.perfetto.as_deref())?;

        // `with_tokio()` is required because the file reader spawns async
        // I/O tasks on the tokio runtime.
        let session = VortexSession::default().with_tokio();
        register_cuda_layout(&session);

        // PERF: Use the default (non-tracing) launch strategy.
        // `TracingLaunchStrategy` synchronises the GPU stream on every kernel
        // launch to read back event timestamps, adding hundreds of ms of
        // stalls across thousands of launches.
        let mut cuda_ctx = CudaSession::create_execution_ctx(&session)?;

        let path = PathBuf::from(&cli.source);

        // Memory-map the file and register with CUDA for zero-copy GPU access.
        // cuMemHostRegister pins the mmap'd pages so the GPU can read them
        // directly over NVLink-C2C without ATS page faults, and without the
        // ~47% overhead of pread-ing into pinned staging buffers.
        let registered_mmap = Arc::new(RegisteredMmap::new(&path, cuda_ctx.stream().context())?);

        let path = PathBuf::from(&cli.source);
        let file_size = std::fs::metadata(&path)?.len();
        let file_mb = file_size as f64 / (1024.0 * 1024.0);

        // ---- Warmup iteration (untimed) ---------------------------------
        // Triggers CUDA JIT compilation and warms OS page cache / file
        // metadata caches so that timed iterations measure steady state.
        eprintln!("Warmup...");
        run_one_scan(&session, &registered_mmap, &mut cuda_ctx).await?;
        eprintln!("Warmup done");

        // ---- Timed iterations -------------------------------------------
        // Each iteration performs the full pipeline from scratch:
        //   1. Open file & read footer
        //   2. Scan batches into memory (I/O + deserialization)
        //   3. Build dynamic-dispatch plans (encoding tree walk + pointer
        //      resolution — zero-copy host ptrs on GH200, H2D copy elsewhere)
        //   4. Launch fused GPU decompression kernels
        //   5. Synchronize stream
        let mut iteration_times = Vec::with_capacity(cli.iterations);

        for iteration in 0..cli.iterations {
            let start = Instant::now();

            run_one_scan(&session, &registered_mmap, &mut cuda_ctx).await?;

            let elapsed = start.elapsed();
            iteration_times.push(elapsed);
            tracing::info!(
                "Iteration {}/{}: {:.3}s",
                iteration + 1,
                cli.iterations,
                elapsed.as_secs_f64()
            );
        }

        // ---- Results ----------------------------------------------------
        let best = iteration_times
            .iter()
            .copied()
            .min()
            .ok_or_else(|| vortex::error::vortex_err!("no iterations completed"))?;
        let best_throughput = file_mb / best.as_secs_f64();

        eprintln!();
        eprintln!("=== Benchmark Results ===");
        eprintln!("Source:     {}", cli.source);
        eprintln!("Iterations: {}", cli.iterations);
        eprintln!("File size:  {:.2} MB", file_mb);
        eprintln!(
            "Best:       {:.3}s = {best_throughput:.2} MB/s",
            best.as_secs_f64()
        );

        Ok(())
    }

    // =====================================================================
    // One full scan: open → read → plan → launch → (caller synchronizes)
    // =====================================================================

    // =================================================================
    // RegisteredMmap: mmap + cuMemHostRegister for zero-copy GPU access
    // =================================================================

    /// A memory-mapped file whose pages are registered with CUDA via
    /// `cuMemHostRegister`.
    ///
    /// Registration pins the pages so the GPU can access them directly
    /// over NVLink-C2C without ATS page faults (unlike plain mmap) and
    /// without the CPU-side `pread` copy overhead (unlike pinned staging
    /// buffers). On drop, the pages are unregistered.
    struct RegisteredMmap {
        mmap: memmap2::Mmap,
        _ctx: Arc<cudarc::driver::CudaContext>,
    }

    // SAFETY: The mmap'd memory is registered with CUDA and remains valid
    // for the lifetime of this struct. The underlying Mmap is Send.
    unsafe impl Sync for RegisteredMmap {}

    impl RegisteredMmap {
        fn new(
            path: impl AsRef<Path>,
            ctx: &Arc<cudarc::driver::CudaContext>,
        ) -> VortexResult<Self> {
            let file = File::open(path.as_ref())?;
            let mmap = unsafe {
                memmap2::Mmap::map(&file)
                    .map_err(|e| vortex::error::vortex_err!("mmap failed: {e}"))?
            };

            // Register with CUDA so the driver can pin and track these pages.
            // PORTABLE: accessible from any CUDA context.
            // READ_ONLY: hint to the driver that GPU will only read.
            ctx.bind_to_thread()
                .map_err(|e| vortex::error::vortex_err!("bind CUDA context: {e}"))?;
            let flags = cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE;
            // SAFETY: cuMemHostRegister takes a *mut but only reads (we don't
            // pass DEVICEMAP). The mmap is valid for the lifetime of this struct.
            let result = unsafe {
                cudarc::driver::sys::cuMemHostRegister_v2(
                    mmap.as_ptr() as usize as *mut std::ffi::c_void,
                    mmap.len(),
                    flags,
                )
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                vortex::error::vortex_bail!("cuMemHostRegister failed: {:?}", result);
            }

            Ok(Self {
                mmap,
                _ctx: Arc::clone(ctx),
            })
        }
    }

    /// Owner wrapper for `Bytes::from_owner` that holds an `Arc<RegisteredMmap>`
    /// and exposes the mmap'd data as `&[u8]`.
    struct MmapBytesOwner {
        mmap: Arc<RegisteredMmap>,
    }

    impl AsRef<[u8]> for MmapBytesOwner {
        fn as_ref(&self) -> &[u8] {
            &self.mmap.mmap
        }
    }

    impl Drop for RegisteredMmap {
        fn drop(&mut self) {
            let _ = self._ctx.bind_to_thread();
            // SAFETY: pointer was registered in `new` and is still valid.
            unsafe {
                cudarc::driver::sys::cuMemHostUnregister(
                    self.mmap.as_ptr() as usize as *mut std::ffi::c_void
                );
            }
        }
    }

    /// Execute one complete scan of the file through the GPU.
    ///
    /// Returns after all kernel launches have been enqueued (but not
    /// necessarily completed — the caller must synchronize the stream).
    async fn run_one_scan(
        session: &VortexSession,
        registered_mmap: &Arc<RegisteredMmap>,
        cuda_ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<()> {
        // 1. Open from the CUDA-registered mmap — zero-copy, no pread overhead.
        let owner = MmapBytesOwner {
            mmap: Arc::clone(registered_mmap),
        };
        let buffer = ByteBuffer::from(Bytes::from_owner(owner));
        let gpu_file = session.open_options().open_buffer(buffer)?;

        // 2. Scan with tuned splits for parallelism (~58 splits distributes
        //    deserialization across tokio workers). Collect all batches so
        //    GPU-referenced host pointers stay alive until after stream sync.
        let mut batches = gpu_file
            .scan()?
            .with_split_by(SplitBy::RowCount(2_500_000))
            .into_array_stream()?;

        let mut keep_alive_batches: Vec<vortex::array::ArrayRef> = Vec::new();

        while let Some(batch) = batches.next().await.transpose()? {
            scan_batch(&batch, cuda_ctx)?;
            keep_alive_batches.push(batch);
        }

        // 3. Sync while all batch arrays (and their mmap-backed buffers)
        //    are still alive.
        cuda_ctx.synchronize_stream()?;
        drop(keep_alive_batches);

        Ok(())
    }

    /// Build dynamic-dispatch plans for every primitive column in a struct
    /// batch and launch the fused GPU decompression kernels.
    ///
    /// Each column is dispatched as a single `dynamic_dispatch_multi` kernel
    /// launch. Processing columns sequentially pipelines plan building with
    /// GPU execution: the GPU works on column N while the CPU builds plans
    /// for column N+1.
    fn scan_batch(batch: &vortex::array::ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<()> {
        let struct_arr = batch
            .as_opt::<StructVTable>()
            .ok_or_else(|| vortex::error::vortex_err!("Expected struct batch"))?;
        let fields: Vec<_> = struct_arr.clone().into_fields();

        for field in &fields {
            let ptype = match field.dtype() {
                DType::Primitive(p, _) => *p,
                _ => continue, // skip non-primitive columns
            };

            if field.encoding_id() == ChunkedVTable::ID {
                let chunked = field
                    .clone()
                    .try_into::<ChunkedVTable>()
                    .map_err(|e| vortex::error::vortex_err!("ChunkedVTable cast: {e}"))?;
                let chunks: Vec<_> = chunked.chunks().to_vec();
                build_and_launch_column(&chunks, ptype, ctx)?;
            } else {
                build_and_launch_column(std::slice::from_ref(field), ptype, ctx)?;
            }
        }

        Ok(())
    }

    /// Build plans for all chunks of a column, embed output pointers,
    /// batch-upload them, and dispatch everything in a single kernel launch.
    ///
    /// Uses the `dynamic_dispatch_multi` kernel with a 2D grid:
    /// `blockIdx.y` selects the chunk, `blockIdx.x` the block within it.
    /// This collapses N kernel launches + N allocs into 1 launch + 2 allocs.
    fn build_and_launch_column(
        chunks: &[vortex::array::ArrayRef],
        ptype: PType,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Phase 1: Build all plans on the host.
        let mut plans: Vec<DynamicDispatchPlan> = Vec::with_capacity(chunks.len());
        let mut chunk_lens: Vec<usize> = Vec::with_capacity(chunks.len());
        let mut all_keep_alive: Vec<BufferHandle> = Vec::new();

        for chunk in chunks {
            let (plan, keep_alive) = match dynamic_dispatch::build_plan(chunk, ctx) {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!(error = %e, "plan build failed, skipping chunk");
                    continue;
                }
            };
            plans.push(plan);
            chunk_lens.push(chunk.len());
            all_keep_alive.extend(keep_alive);
        }

        if plans.is_empty() {
            return Ok(());
        }

        let kernel_ptype = unsigned_ptype(ptype);
        let smem_bytes = smem_for_ptype(&plans[0], ptype);

        // Phase 2: Allocate one output buffer for the entire column.
        let total_output_elems: usize = chunk_lens
            .iter()
            .map(|len| len.next_multiple_of(1024))
            .sum();
        let output_slice = ctx.device_alloc::<u32>(total_output_elems)?;
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_view = output_buf.as_view::<u32>();
        let (output_base_ptr, record_output) = output_view.device_ptr(ctx.stream());

        // Phase 3: Embed output_ptr and array_len into each plan so the
        // multi kernel can read them from the plan array directly.
        let mut output_offset: u64 = 0;
        for (plan, &chunk_len) in plans.iter_mut().zip(chunk_lens.iter()) {
            plan.output_ptr = output_base_ptr + output_offset * 4; // u32 = 4 bytes
            plan.array_len = chunk_len as u64;
            output_offset += chunk_len.next_multiple_of(1024) as u64;
        }

        // Phase 4: Batch-upload all plans to the device in one copy.
        let device_plans = ctx
            .stream()
            .clone_htod(&plans)
            .map_err(|e| vortex::error::vortex_err!("batch copy plans to device: {e}"))?;
        let (plans_ptr, record_plans) = device_plans.device_ptr(ctx.stream());

        // Phase 5: Load the multi-chunk kernel from the `dynamic_dispatch` module.
        // The function name is `dynamic_dispatch_multi_{ptype}`, so we use
        // the suffixes ["multi", "{ptype}"] with module name "dynamic_dispatch".
        let ptype_str = kernel_ptype.to_string();
        let cuda_function: CudaFunction =
            ctx.load_function_with_suffixes("dynamic_dispatch", &["multi", &ptype_str])?;

        // grid.x = max blocks any single chunk needs
        // grid.y = number of chunks
        #[allow(clippy::cast_possible_truncation)]
        let max_blocks = chunk_lens
            .iter()
            .map(|len| len.div_ceil(2048) as u32)
            .max()
            .unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        let num_chunks = plans.len() as u32;

        let mut builder = ctx.stream().launch_builder(&cuda_function);
        builder.arg(&plans_ptr);

        let config = LaunchConfig {
            grid_dim: (max_blocks, num_chunks, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        unsafe {
            builder
                .launch(config)
                .map_err(|e| vortex::error::vortex_err!("kernel launch failed: {e}"))?;
        }

        drop((record_output, record_plans, all_keep_alive));

        Ok(())
    }

    // ---- Helpers --------------------------------------------------------

    fn smem_for_ptype(plan: &DynamicDispatchPlan, ptype: PType) -> u32 {
        match ptype {
            PType::U8 | PType::I8 => plan.shared_mem_bytes::<u8>(),
            PType::U16 | PType::I16 => plan.shared_mem_bytes::<u16>(),
            PType::U32 | PType::I32 | PType::F32 => plan.shared_mem_bytes::<u32>(),
            PType::U64 | PType::I64 | PType::F64 => plan.shared_mem_bytes::<u64>(),
            _ => plan.shared_mem_bytes::<u32>(),
        }
    }

    /// Map signed ptypes to unsigned equivalents — the `dynamic_dispatch`
    /// kernel is only instantiated for unsigned integer types but the bit
    /// representation is identical for signed/unsigned of the same width.
    fn unsigned_ptype(ptype: PType) -> PType {
        match ptype {
            PType::I8 => PType::U8,
            PType::I16 => PType::U16,
            PType::I32 | PType::F32 => PType::U32,
            PType::I64 | PType::F64 => PType::U64,
            other => other,
        }
    }
}
