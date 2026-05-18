// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executor that links a Copy-and-Patch trampoline against the stencils
//! required by a `Plan` and launches it.
//!
//! Layout:
//!   1. Read the trampoline PTX + each stencil PTX from `kernels/gen/`.
//!   2. Call `cuLink*` to produce a single cubin.
//!   3. Load the cubin via `cuModuleLoadData` and resolve the kernel name.
//!   4. Launch one warp per 1024-element chunk.
//!
//! The cubin is cached by plan signature so subsequent launches with the
//! same shape skip the link step. Cache keys deliberately use the stencil
//! module-name set (not the runtime constants) — `f`, `e`, and the post-op
//! constant flow in as kernel args.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::utils::aliases::dash_map::DashMap;

use super::linker::LinkInput;
use super::linker::LinkInputKind;
use super::linker::link_modules;
use super::plan::Plan;
use super::plan::PostOp;
use crate::executor::CudaExecutionCtx;

/// Output of a Copy-and-Patch launch.
///
/// Arith plans produce an `f32` device buffer rounded up to the next
/// 1024-element chunk; filter plans produce a `u8` mask of the same shape.
/// Callers slice off the padding using the original `array_len`.
pub enum CopyPatchOutput {
    Arith(CudaSlice<f32>),
    Filter(CudaSlice<u8>),
}

/// Cache of linked modules + trampoline functions, keyed by the set of
/// stencil module names that went into the link.
#[derive(Default)]
pub struct CopyPatchCache {
    entries: DashMap<String, Arc<LinkedTrampoline>>,
}

struct LinkedTrampoline {
    #[allow(dead_code)] // retained to keep the cubin alive for the function handle
    module: Arc<CudaModule>,
    function: CudaFunction,
}

/// Executor for Copy-and-Patch plans.
///
/// Reuses the surrounding `CudaExecutionCtx` for its stream, device
/// allocation, and CUDA context. The cache lives on the executor so the
/// caller decides its scope.
pub struct CopyPatchExecutor {
    cache: CopyPatchCache,
    kernels_dir: PathBuf,
}

impl CopyPatchExecutor {
    /// Create a fresh executor. PTX files are read from the directory
    /// indicated by the `VORTEX_CUDA_KERNELS_DIR` environment variable if
    /// set, else from the build-time-baked path.
    pub fn new() -> Self {
        let kernels_dir = std::env::var("VORTEX_CUDA_KERNELS_DIR")
            .unwrap_or_else(|_| env!("VORTEX_CUDA_KERNELS_DIR").to_string());
        Self {
            cache: CopyPatchCache::default(),
            kernels_dir: PathBuf::from(kernels_dir),
        }
    }

    /// Number of cached linked modules. Useful for tests that want to
    /// verify the cache is hit on a repeat plan.
    pub fn cache_size(&self) -> usize {
        self.cache.entries.len()
    }

    /// Link the stencils for `plan` and cache the resulting cubin without
    /// launching. Useful for benchmarks that want to separate the link
    /// latency (the headline cost of Copy-and-Patch) from the launch
    /// latency, and for application startup paths that want to pay link
    /// cost up-front rather than on the first query.
    pub fn warm_up(&self, ctx: &CudaExecutionCtx, plan: &Plan) -> VortexResult<()> {
        if plan.bit_width > 32 {
            vortex_bail!(
                "Copy-and-Patch u32 stencil supports bit_width <= 32, got {}",
                plan.bit_width
            );
        }
        let context = Arc::<CudaContext>::clone(ctx.stream().context());
        self.get_or_link(&context, plan).map(|_| ())
    }

    /// Link (or fetch from cache) the trampoline kernel needed by `plan`.
    fn get_or_link(
        &self,
        ctx: &Arc<CudaContext>,
        plan: &Plan,
    ) -> VortexResult<Arc<LinkedTrampoline>> {
        let mut modules = Vec::with_capacity(4);
        modules.push(plan.trampoline_module());
        modules.extend(plan.stencil_modules());
        let cache_key = modules.join("+");

        if let Some(entry) = self.cache.entries.get(&cache_key) {
            return Ok(Arc::clone(entry.value()));
        }

        // Read all PTX inputs from disk.
        let ptx_bufs: Vec<Vec<u8>> = modules
            .iter()
            .map(|m| read_ptx(&self.kernels_dir, m))
            .collect::<VortexResult<_>>()?;

        let inputs: Vec<LinkInput<'_>> = modules
            .iter()
            .zip(&ptx_bufs)
            .map(|(name, data)| LinkInput {
                name,
                kind: LinkInputKind::Ptx,
                data,
            })
            .collect();

        let linked = link_modules(ctx, &inputs)?;
        let module = ctx
            .load_module(linked)
            .map_err(|e| vortex_err!("load_module(linked cubin) failed: {e}"))?;

        let function = module
            .load_function(plan.trampoline_entry())
            .map_err(|e| vortex_err!("load_function({}): {e}", plan.trampoline_entry()))?;

        let entry = Arc::new(LinkedTrampoline { module, function });
        self.cache.entries.insert(cache_key, Arc::clone(&entry));
        Ok(entry)
    }

    /// Launch the trampoline against a packed input that already lives on
    /// the device. The caller is responsible for ensuring `packed_input` is
    /// laid out as raw FastLanes-packed `u32` words (1024 / 32 = 32 lanes,
    /// `bit_width` rows of u32 per chunk).
    ///
    /// `array_len` is the logical length (chunks need not be full). The
    /// output buffer is allocated with capacity rounded up to a chunk.
    pub fn launch<'a>(
        &self,
        ctx: &mut CudaExecutionCtx,
        plan: &Plan,
        packed_input: CudaView<'a, u32>,
        array_len: usize,
    ) -> VortexResult<CopyPatchOutput> {
        if array_len == 0 {
            vortex_bail!("Copy-and-Patch launch requires array_len > 0");
        }
        if plan.bit_width > 32 {
            vortex_bail!(
                "Copy-and-Patch u32 stencil supports bit_width <= 32, got {}",
                plan.bit_width
            );
        }

        let chunks = array_len.div_ceil(1024);
        let enc_stride_words: u32 = u32::from(plan.bit_width) * 32;
        let array_len_u64 = array_len as u64;

        let context = Arc::<CudaContext>::clone(ctx.stream().context());
        let trampoline = self.get_or_link(&context, plan)?;

        // 32 threads per block (one FastLanes warp). Chunks are independent.
        let cfg = LaunchConfig {
            grid_dim: (u32::try_from(chunks)?, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        match plan.post {
            PostOp::Arith { c, .. } => {
                let out = ctx.device_alloc::<f32>(chunks * 1024)?;
                let out_view = out.as_view();
                let f = plan.f;
                let e = plan.e;
                ctx.launch_kernel_config(&trampoline.function, cfg, array_len, |args| {
                    args.arg(&packed_input)
                        .arg(&out_view)
                        .arg(&array_len_u64)
                        .arg(&enc_stride_words)
                        .arg(&f)
                        .arg(&e)
                        .arg(&c);
                })?;
                Ok(CopyPatchOutput::Arith(out))
            }
            PostOp::Filter { c, .. } => {
                let out = ctx.device_alloc::<u8>(chunks * 1024)?;
                let out_view = out.as_view();
                let f = plan.f;
                let e = plan.e;
                ctx.launch_kernel_config(&trampoline.function, cfg, array_len, |args| {
                    args.arg(&packed_input)
                        .arg(&out_view)
                        .arg(&array_len_u64)
                        .arg(&enc_stride_words)
                        .arg(&f)
                        .arg(&e)
                        .arg(&c);
                })?;
                Ok(CopyPatchOutput::Filter(out))
            }
        }
    }
}

impl Default for CopyPatchExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn read_ptx(kernels_dir: &Path, module_name: &str) -> VortexResult<Vec<u8>> {
    let path = kernels_dir.join(format!("{module_name}.ptx"));
    std::fs::read(&path).map_err(|e| {
        vortex_err!(
            "failed to read Copy-and-Patch PTX module {} at {}: {e}",
            module_name,
            path.display()
        )
    })
}
