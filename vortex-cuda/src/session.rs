// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use vortex::array::ArrayId;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::SessionExt;
use vortex::session::SessionGuard;
use vortex::session::SessionVar;
use vortex::utils::aliases::dash_map::DashMap;
use vortex_cuda_driver::CudaContext;

use crate::ExportDeviceArray;
use crate::arrow::CanonicalDeviceArrayExport;
use crate::executor::CudaExecute;
pub use crate::executor::CudaExecutionCtx;
use crate::initialize_cuda;
use crate::kernel::KernelLoader;
use crate::pinned::PinnedByteBufferPool;
use crate::stream::VortexCudaStream;
use crate::stream_pool::VortexCudaStreamPool;

/// Default maximum number of streams in the pool.
const DEFAULT_STREAM_POOL_CAPACITY: usize = 4;

/// Shares the context across live default sessions without keeping it alive permanently.
/// The last owner dropping releases our context reference and frees its owned memory pool;
/// a later session initializes them again.
static DEFAULT_CUDA_CONTEXT: Mutex<Weak<CudaContext>> = Mutex::new(Weak::new());

// Default-context sharing is session policy, not driver policy. Taking the cache explicitly lets
// lifetime tests exercise it without interference from other tests using the global cache.
fn get_or_init_default_context(cache: &Mutex<Weak<CudaContext>>) -> VortexResult<Arc<CudaContext>> {
    // Serialize creation as well as lookup so concurrent sessions share one pool.
    let mut cached = cache.lock();
    if let Some(context) = cached.upgrade() {
        return Ok(context);
    }
    let context = CudaContext::new(0)
        .map_err(|err| vortex_err!("failed to initialize CUDA device 0: {err}"))?;
    *cached = Arc::downgrade(&context);
    Ok(context)
}

/// Arrow Device layout used when exporting variable-length UTF-8 and binary arrays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VarBinExportLayout {
    /// Offset-based Arrow `Utf8`/`Binary` with one contiguous values buffer.
    #[default]
    VarBin,
    /// Arrow `Utf8View`/`BinaryView` with 16-byte views and variadic data buffers.
    VarBinView,
}

/// Controls whether Arrow Device exports keep dictionary encoding or fully decode it,
/// including dictionaries nested inside structs and lists.
///
/// For example, indices `[0, 1, 0]` and dictionary values `["apple", "pear"]` export as:
///
/// - [`Preserve`](Self::Preserve): separate index and dictionary-value arrays, with an Arrow
///   dictionary type.
/// - [`Decode`](Self::Decode): the plain string array `["apple", "pear", "apple"]`, with no
///   dictionary indices or dictionary child.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DictionaryExport {
    /// Keep separate index and dictionary-value arrays rather than expanding repeated values.
    /// The Arrow schema retains the dictionary type, including its index type.
    #[default]
    Preserve,
    /// Expand dictionaries into plain values for a stable schema across batches.
    /// May increase device memory use; device-resident inputs require CUDA decoding support.
    Decode,
}

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
/// Holds the CUDA context for all GPU operations and caches compiled PTX modules.
/// Default sessions share a lazily initialized context and its device memory pool, but have
/// independent kernel registries, stream pools, and export settings.
#[derive(Clone, Debug)]
pub struct CudaSession {
    context: Arc<CudaContext>,
    kernels: Arc<DashMap<ArrayId, &'static dyn CudaExecute>>,
    export_device_array: Arc<dyn ExportDeviceArray>,
    varbin_export_layout: VarBinExportLayout,
    dictionary_export: DictionaryExport,
    kernel_loader: Arc<KernelLoader>,
    stream_pool: Arc<VortexCudaStreamPool>,
    pinned_buffer_pool: Arc<PinnedByteBufferPool>,
}

impl CudaSession {
    /// Creates a CUDA session with an explicit context and default stream pool capacity.
    ///
    /// Uses the supplied context as-is, bypassing default CUDA context sharing. Prefer
    /// [`Self::try_default`] to share the default context and its Vortex-owned memory pool.
    pub fn with_context(context: Arc<CudaContext>) -> Self {
        let stream_pool = Arc::new(VortexCudaStreamPool::new(
            Arc::clone(&context),
            DEFAULT_STREAM_POOL_CAPACITY,
        ));
        let pinned_buffer_pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(&context)));
        Self {
            context,
            kernels: Arc::new(DashMap::default()),
            kernel_loader: Arc::new(KernelLoader::new()),
            export_device_array: Arc::new(CanonicalDeviceArrayExport),
            varbin_export_layout: VarBinExportLayout::default(),
            dictionary_export: DictionaryExport::default(),
            stream_pool,
            pinned_buffer_pool,
        }
    }

    /// Sets the stream pool capacity without changing the CUDA context or its memory pool.
    ///
    /// Replaces this session's stream pool. Existing streams and other session clones keep
    /// using the previous pool's streams.
    ///
    /// # Panics
    ///
    /// Panics if `stream_pool_capacity` is zero.
    pub fn with_stream_pool_capacity(mut self, stream_pool_capacity: usize) -> Self {
        self.stream_pool = Arc::new(VortexCudaStreamPool::new(
            Arc::clone(&self.context),
            stream_pool_capacity,
        ));
        self
    }

    /// Selects the Arrow Device layout for variable-length UTF-8 and binary exports.
    pub fn with_varbin_export_layout(mut self, layout: VarBinExportLayout) -> Self {
        self.varbin_export_layout = layout;
        self
    }

    /// Returns the Arrow Device layout used for variable-length UTF-8 and binary exports.
    pub fn varbin_export_layout(&self) -> VarBinExportLayout {
        self.varbin_export_layout
    }

    /// Selects whether Arrow Device exports preserve or decode dictionaries.
    pub fn with_dictionary_export(mut self, policy: DictionaryExport) -> Self {
        self.dictionary_export = policy;
        self
    }

    /// Returns the dictionary policy used for Arrow Device exports.
    pub fn dictionary_export(&self) -> DictionaryExport {
        self.dictionary_export
    }

    /// Creates a default CUDA session using device 0, with all GPU array kernels preloaded.
    ///
    /// The context and its fixed Vortex-owned memory pool are shared with other live default
    /// sessions. They are released when the last owner drops, including streams and buffers that
    /// outlive their
    /// sessions. A later session initializes a new context. Initialization failures are not cached.
    ///
    /// The memory pool is shared, but operation errors return directly to each caller. Use
    /// [`Self::with_context`] with a separate wrapper for independent allocation policy.
    /// Separate wrappers created with [`CudaContext::new`] still use the same native primary
    /// context; use [`CudaContext::new_non_primary`] when a separate native context is required.
    ///
    /// Unlike [`Default::default`], this returns an error instead of panicking when CUDA cannot be
    /// initialized.
    pub fn try_default() -> VortexResult<Self> {
        let context = get_or_init_default_context(&DEFAULT_CUDA_CONTEXT)?;
        let this = Self::with_context(context);
        initialize_cuda(&this);
        Ok(this)
    }

    /// Creates a new CUDA execution context.
    pub fn create_execution_ctx(
        vortex_session: &vortex::session::VortexSession,
    ) -> VortexResult<CudaExecutionCtx> {
        let stream = vortex_session.cuda_session().stream()?;
        Ok(CudaExecutionCtx::new(
            stream,
            vortex_session.create_execution_ctx(),
        ))
    }

    /// Returns a CUDA stream from the pool.
    ///
    /// The pool reuses existing streams in round-robin fashion.
    pub fn stream(&self) -> VortexResult<VortexCudaStream> {
        self.stream_pool.stream()
    }

    /// Returns the session-scoped pool used for staging file reads in pinned host memory.
    pub fn pinned_buffer_pool(&self) -> &Arc<PinnedByteBufferPool> {
        &self.pinned_buffer_pool
    }

    /// Registers CUDA support for an array encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to register support for
    /// * `executor` - A static reference to the CUDA support implementation
    pub fn register_kernel(
        &self,
        array_id: impl Into<ArrayId>,
        executor: &'static dyn CudaExecute,
    ) {
        self.kernels.insert(array_id.into(), executor);
    }

    /// Retrieves the CUDA support implementation for an encoding, if registered.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to look up
    pub fn kernel(&self, array_id: &ArrayId) -> Option<&'static dyn CudaExecute> {
        self.kernels.get(array_id).map(|entry| *entry.value())
    }

    /// Loads a CUDA kernel function by module name and type suffixes.
    ///
    /// This is a lower-level version of `load_function` that accepts string suffixes
    /// directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// The kernel name is generated as `{module_name}_{suffix[0]}_{suffix[1]}...`
    ///
    /// # Arguments
    ///
    /// * `module_name` - Kernel source name without the `.cu` extension
    /// * `type_suffixes` - List of type suffix strings to generate kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if the module was not embedded or the kernel cannot be loaded.
    pub fn load_function_with_suffixes(
        &self,
        module_name: &str,
        type_suffixes: &[&str],
    ) -> VortexResult<vortex_cuda_driver::CudaFunction> {
        self.kernel_loader
            .load_function(module_name, type_suffixes, &self.context)
    }

    /// Get a handle to the exporter that converts Vortex arrays to `ArrowDeviceArray`.
    pub fn export_device_array(&self) -> &Arc<dyn ExportDeviceArray> {
        &self.export_device_array
    }
}

impl Default for CudaSession {
    /// Creates a default CUDA session using device 0, with all GPU array kernels preloaded.
    ///
    /// # Panics
    ///
    /// Panics if CUDA device 0 cannot be initialized.
    #[expect(clippy::expect_used)]
    fn default() -> Self {
        Self::try_default().expect("Failed to initialize CUDA device 0")
    }
}

impl SessionVar for CudaSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing the CUDA session from a Vortex session.
pub trait CudaSessionExt: SessionExt {
    /// Returns the CUDA session.
    fn cuda_session(&self) -> SessionGuard<'_, CudaSession> {
        self.get::<CudaSession>()
    }
}
impl<S: SessionExt> CudaSessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::Weak;
    use std::thread;

    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex_cuda_driver::CudaContext;

    use super::CudaSession;
    use super::get_or_init_default_context;

    #[crate::test]
    fn default_sessions_share_context_but_not_session_state() -> VortexResult<()> {
        let first = CudaSession::try_default()?;
        let second = CudaSession::try_default()?;

        assert!(Arc::ptr_eq(&first.context, &second.context));
        assert_eq!(first.context.mem_pool(), second.context.mem_pool());
        assert!(!Arc::ptr_eq(&first.kernels, &second.kernels));
        assert!(!Arc::ptr_eq(&first.stream_pool, &second.stream_pool));
        assert!(!Arc::ptr_eq(
            &first.pinned_buffer_pool,
            &second.pinned_buffer_pool
        ));
        Ok(())
    }

    #[crate::test]
    fn explicit_context_is_preserved_when_configuring_stream_pool() -> VortexResult<()> {
        let default = CudaSession::try_default()?;
        let context = CudaContext::new(0)
            .map_err(|err| vortex_err!("failed to initialize CUDA device 0: {err}"))?;
        let original = CudaSession::with_context(Arc::clone(&context));
        let original_capacity = original.stream_pool.max_capacity();
        let configured = original.clone().with_stream_pool_capacity(2);

        assert_eq!(configured.stream_pool.max_capacity(), 2);
        assert_eq!(original.stream_pool.max_capacity(), original_capacity);
        assert!(!Arc::ptr_eq(&configured.stream_pool, &original.stream_pool));
        assert!(Arc::ptr_eq(&configured.context, &context));
        assert!(Arc::ptr_eq(configured.stream()?.context(), &context));
        assert!(!Arc::ptr_eq(&configured.context, &default.context));
        Ok(())
    }

    #[rstest]
    #[case::stream(false)]
    #[case::buffer(true)]
    #[crate::test]
    fn cached_context_lives_until_last_resource_drops(
        #[case] keep_buffer: bool,
    ) -> VortexResult<()> {
        // A local cache isolates lifetime assertions from other tests' default sessions.
        let cache = Mutex::new(Weak::new());
        let session = CudaSession::with_context(get_or_init_default_context(&cache)?);
        let weak = Arc::downgrade(&session.context);
        let stream = session.stream()?;
        let buffer = keep_buffer
            .then(|| stream.device_alloc::<u8>(1))
            .transpose()?;

        drop(session);
        assert!(weak.upgrade().is_some());
        drop(stream);
        assert_eq!(weak.upgrade().is_some(), keep_buffer);
        drop(buffer);
        assert!(weak.upgrade().is_none());

        let recreated = get_or_init_default_context(&cache)?;
        assert!(!Weak::ptr_eq(&weak, &Arc::downgrade(&recreated)));
        Ok(())
    }

    #[crate::test]
    fn concurrent_initialization_shares_context() -> VortexResult<()> {
        let cache = Mutex::new(Weak::new());
        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            let worker = scope.spawn(|| {
                barrier.wait();
                get_or_init_default_context(&cache)
            });
            barrier.wait();
            let context = get_or_init_default_context(&cache)?;
            let other = worker.join().expect("context worker panicked")?;
            assert!(Arc::ptr_eq(&context, &other));
            Ok(())
        })
    }
}
