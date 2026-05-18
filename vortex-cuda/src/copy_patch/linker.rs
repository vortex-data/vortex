// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Safe wrapper around the CUDA driver linker (`cuLink*`).
//!
//! The driver linker assembles multiple PTX or cubin inputs into a single
//! cubin image that can be loaded as a CUDA module. Copy-and-Patch uses it
//! to stitch a thin trampoline together with pre-compiled stencil modules,
//! resolving cross-module `__device__` function references at link time
//! without re-running the C++ frontend (NVRTC) or the optimizer.
//!
//! ptxas still runs over PTX inputs — typically 1–10 ms for the small
//! fragments we feed it, with caching across link invocations of the same
//! stencil set. To eliminate ptxas entirely, stencils can be promoted to
//! pre-built cubin and added with `CU_JIT_INPUT_CUBIN`.

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use cudarc::nvrtc::Ptx;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

/// A PTX or cubin module to feed into the link step.
pub struct LinkInput<'a> {
    pub name: &'a str,
    pub kind: LinkInputKind,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub enum LinkInputKind {
    /// Text PTX. ptxas runs over it during link.
    Ptx,
    /// Pre-compiled cubin for a specific SM. No ptxas pass at link time.
    Cubin,
}

impl LinkInputKind {
    fn as_sys(self) -> sys::CUjitInputType {
        match self {
            Self::Ptx => sys::CUjitInputType::CU_JIT_INPUT_PTX,
            Self::Cubin => sys::CUjitInputType::CU_JIT_INPUT_CUBIN,
        }
    }
}

/// Link a set of PTX/cubin inputs into a single cubin image.
///
/// Returns a `cudarc` `Ptx` wrapping the cubin bytes, which can be passed
/// straight to `CudaContext::load_module`. The cubin is held in a Rust
/// `Vec<u8>` so it can be reloaded without re-linking.
///
/// The first input by convention is the trampoline kernel; stencils follow.
/// The link order doesn't matter for symbol resolution, but it does matter
/// for the order of error messages if one of the inputs fails to parse.
pub fn link_modules(_ctx: &Arc<CudaContext>, inputs: &[LinkInput<'_>]) -> VortexResult<Ptx> {
    if inputs.is_empty() {
        return Err(vortex_err!("link_modules: at least one input required"));
    }

    // Create the link state. We currently pass no JIT options; production
    // code would forward CU_JIT_TARGET (SM version), CU_JIT_OPTIMIZATION_LEVEL,
    // and CU_JIT_LOG_VERBOSE here.
    let mut state = MaybeUninit::<sys::CUlinkState>::uninit();
    unsafe {
        sys::cuLinkCreate_v2(0, ptr::null_mut(), ptr::null_mut(), state.as_mut_ptr())
            .result()
            .map_err(|e| vortex_err!("cuLinkCreate failed: {e}"))?;
    }
    // SAFETY: cuLinkCreate succeeded, so the state is initialized.
    let state = unsafe { state.assume_init() };

    // Guard that destroys the link state on every exit path.
    struct LinkGuard(sys::CUlinkState);
    impl Drop for LinkGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = sys::cuLinkDestroy(self.0);
            }
        }
    }
    let _guard = LinkGuard(state);

    for input in inputs {
        let name_c = CString::new(input.name)
            .map_err(|e| vortex_err!("link input name contains nul byte: {e}"))?;
        // cuLinkAddData takes a non-const pointer but does not mutate the
        // contents — it copies the bytes into its own internal buffer.
        #[expect(
            clippy::as_ptr_cast_mut,
            reason = "input.data is borrowed immutably; cuLinkAddData copies it but signature is *mut"
        )]
        let data_ptr = input.data.as_ptr() as *mut std::ffi::c_void;
        let res = unsafe {
            sys::cuLinkAddData_v2(
                state,
                input.kind.as_sys(),
                data_ptr,
                input.data.len(),
                name_c.as_ptr(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
            .result()
        };
        res.map_err(|e| vortex_err!("cuLinkAddData({}) failed: {e}", input.name))?;
    }

    let mut cubin_ptr: *mut std::ffi::c_void = ptr::null_mut();
    let mut cubin_size: usize = 0;
    unsafe {
        sys::cuLinkComplete(state, &raw mut cubin_ptr, &raw mut cubin_size)
            .result()
            .map_err(|e| vortex_err!("cuLinkComplete failed: {e}"))?;
    }

    if cubin_ptr.is_null() || cubin_size == 0 {
        return Err(vortex_err!("cuLinkComplete returned an empty cubin"));
    }

    // The cubin pointer is owned by the link state and freed when it is
    // destroyed. Copy the bytes out so the cubin survives `LinkGuard::drop`.
    // SAFETY: the driver guarantees the pointer is valid and at least
    // `cubin_size` bytes long until `cuLinkDestroy` is called.
    let cubin: Vec<u8> =
        unsafe { std::slice::from_raw_parts(cubin_ptr as *const u8, cubin_size).to_vec() };

    Ok(Ptx::from_binary(cubin))
}
