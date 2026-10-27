// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loaded device code.

use std::ffi::CString;
use std::sync::Arc;

use crate::context::CudaContext;
use crate::result::DriverError;
use crate::sys;
use crate::sys::CUresult;

/// A loaded cubin, fatbin or PTX module, unloaded when the last function is dropped.
pub struct CudaModule {
    module: sys::CUmodule,
    context: Arc<CudaContext>,
}

// SAFETY: Module handles are usable from any thread.
unsafe impl Send for CudaModule {}
// SAFETY: As above.
unsafe impl Sync for CudaModule {}

impl std::fmt::Debug for CudaModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModule")
            .field("module", &self.module)
            .finish()
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.context.record_err(self.context.make_current());
        // SAFETY: Unloaded once, after every function referencing it was dropped.
        self.context
            .record_err(unsafe { sys::cuModuleUnload(self.module) }.result());
    }
}

impl CudaModule {
    pub(crate) fn load(context: &Arc<CudaContext>, image: &[u8]) -> Result<Arc<Self>, DriverError> {
        context.bind_to_thread()?;
        let mut module = std::ptr::null_mut();
        // SAFETY: The driver reads a complete image; fatbins and cubins carry their own size.
        unsafe { sys::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) }.result()?;
        Ok(Arc::new(Self {
            module,
            context: Arc::clone(context),
        }))
    }

    /// The kernel named `name`.
    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<CudaFunction, DriverError> {
        let name =
            CString::new(name).map_err(|_| DriverError(CUresult::CUDA_ERROR_INVALID_VALUE))?;
        self.context.bind_to_thread()?;
        let mut function = std::ptr::null_mut();
        // SAFETY: The module is loaded; the driver writes the function handle.
        unsafe { sys::cuModuleGetFunction(&raw mut function, self.module, name.as_ptr()) }
            .result()?;
        Ok(CudaFunction {
            function,
            _module: Arc::clone(self),
        })
    }
}

/// A kernel in a loaded module, which it keeps loaded.
#[derive(Clone)]
pub struct CudaFunction {
    function: sys::CUfunction,
    _module: Arc<CudaModule>,
}

// SAFETY: Function handles are usable from any thread.
unsafe impl Send for CudaFunction {}
// SAFETY: As above.
unsafe impl Sync for CudaFunction {}

impl std::fmt::Debug for CudaFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaFunction")
            .field("function", &self.function)
            .finish()
    }
}

impl CudaFunction {
    /// The raw function handle.
    pub fn cu_function(&self) -> sys::CUfunction {
        self.function
    }
}
