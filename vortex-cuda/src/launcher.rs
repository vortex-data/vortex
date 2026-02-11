// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchConfig;
use cudarc::driver::result;
use cudarc::driver::sys;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// A CUDA kernel function.
///
/// Functions are not invoked directly. Instead, they are mapped by a [`Kernel`], which creates
/// a typed launcher.
pub struct Function {
    module: Arc<Module>,
    // NOTE: this does not need its own cleanup logic, that is handled by the Module drop logic.
    inner: sys::CUfunction,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub(crate) fn cu_function(self) -> sys::CUfunction {
        self.inner
    }
}

/// A CUDA module. This maps 1:1 onto a compiled PTX binary.
///
/// Modules contain one or more [`Function`]s, which can be launched on the GPU.
#[derive(Clone, Debug)]
pub struct Module {
    inner: sys::CUmodule,
    context: Arc<CudaContext>,
}

// SAFETY: Module can be sent between threads because we always bind_to_thread().
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Load a module from a path to a PTX file, mapping it into the device associated with
    /// the provided CudaContext.
    pub fn from_ptx(ctx: Arc<CudaContext>, path: impl AsRef<Path>) -> VortexResult<Self> {
        ctx.bind_to_thread()
            .map_err(|e| vortex_err!("bind_to_thread: {e}"))?;

        let path_string = CString::from_str(path.as_ref().to_string_lossy().as_ref())
            .expect("path cannot contain null bytes");

        let module =
            result::module::load(path_string).map_err(|e| vortex_err!("load module: {e}"))?;

        Ok(Self {
            inner: module,
            context: ctx,
        })
    }

    /// Lookup and load the given CUDA kernel from this module.
    ///
    /// If the kernel is not present or there is an error accessing the module, an error is returned.
    pub fn load<K: Kernel>(self: &Arc<Self>, kernel_id: String) -> VortexResult<K> {
        self.context
            .bind_to_thread()
            .map_err(|e| vortex_err!("bind_to_thread: {e}"))?;

        let name = kernel_id.to_string();
        let ffi_name = CString::new(name).expect("name cannot contain null bytes");

        // TODO(aduffy): add caching for function lookup.
        // SAFETY: self.inner points at a valid CUmodule that is still allocated.
        let func = unsafe {
            result::module::get_function(self.inner, ffi_name)
                .map_err(|e| vortex_err!("Module::get_function: {e}"))?
        };

        let function = Function {
            module: Arc::clone(self),
            inner: func,
        };

        Ok(K::new(function))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: points to our struct field.
        let cu_module = unsafe { std::ptr::replace(&raw mut self.inner, std::ptr::null_mut()) };
        unsafe {
            if let Err(e) = result::module::unload(cu_module) {
                tracing::warn!(error = ?e, "failed to unload cuModule");
            }
        }
    }
}

/// A CUDA kernel.
///
/// CUDA kernels are the core GPU compute type. Kernels have arguments that can be set on the host,
/// and then launched to execute massively in parallel on the GPU.
pub trait Kernel {
    /// Type of kernel arguments. These must all be types which are safe to copy onto the device.
    type Args;

    /// Create a new instance of the kernel. Any dynamic information necessary for lookup
    /// can be constructed this way.
    fn new(function: Function) -> Self;

    /// Launch self onto a copy of a launcher.
    ///
    /// This should only be called by the harness
    unsafe fn launch(self, args: Self::Args, launcher: &Arc<dyn Launcher>) -> VortexResult<()>;
}

pub trait Launcher: Debug + Send + Sync + 'static {
    /// Launch a function with a set of arguments.
    ///
    /// # Safety
    ///
    /// The provided arguments pointer needs to point to memory that is valid at the time we are
    /// called.
    ///
    /// If not, this will lead to the CUDA driver accessing invalid memory and causing undefined
    /// behavior.
    unsafe fn launch(
        &self,
        function: Function,
        cfg: LaunchConfig,
        args: Vec<*mut std::ffi::c_void>,
    ) -> VortexResult<()>;
}

/// The default kernel launcher, which launches kernels onto the GPU without tracking any
/// relative timing information.
#[derive(Debug, Clone)]
pub(crate) struct AsyncLauncher {
    stream: Arc<CudaStream>,
}

impl AsyncLauncher {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl Launcher for AsyncLauncher {
    unsafe fn launch(
        &self,
        function: Function,
        cfg: LaunchConfig,
        mut args: Vec<*mut std::ffi::c_void>,
    ) -> VortexResult<()> {
        // IMPORTANT: must bind context to current thread before we launch.
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| vortex_err!("bind_to_thread: {e}"))?;

        // SAFETY: enforced by the caller. See docs on Launcher::launch.
        unsafe {
            result::launch_kernel(
                function.cu_function(),
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                self.stream.cu_stream(),
                args.as_mut(),
            )
            .map_err(|e| vortex_err!("AsyncLauncher::stream: {e}"))
        }
    }
}
