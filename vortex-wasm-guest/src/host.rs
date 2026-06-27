// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Access to the host-provided imports (`vortex_host` module) and the guest allocator.

use vortex_error::VortexResult;
#[cfg(not(target_arch = "wasm32"))]
use vortex_error::vortex_bail;

#[cfg(target_arch = "wasm32")]
mod imports {
    #[link(wasm_import_module = "vortex_host")]
    unsafe extern "C" {
        pub fn vx_decode_child(node_index: i32, out_ptr: i32) -> i32;
        pub fn vx_host_log(ptr: i32, len: i32);
    }
}

/// Ask the host to decode the child array at `node_index` (an index into the root node's children)
/// and return its [`CanonicalMessage`](crate::message) bytes.
#[cfg(target_arch = "wasm32")]
pub fn decode_child(node_index: usize) -> VortexResult<Vec<u8>> {
    let mut out = [0u8; 8];
    let rc = unsafe { imports::vx_decode_child(node_index as i32, out.as_mut_ptr() as i32) };
    if rc != 0 {
        vortex_error::vortex_bail!("host vx_decode_child returned error {rc}");
    }
    let off = u32::from_le_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
    let len = u32::from_le_bytes(out[4..8].try_into().expect("4 bytes")) as usize;
    let slice = unsafe { core::slice::from_raw_parts(off as *const u8, len) };
    Ok(slice.to_vec())
}

/// Host stub used when the SDK is compiled for a non-wasm target (e.g. unit tests). Calling it is
/// a programming error; real kernels run under [`crate::WasmKernel`](../../vortex_wasm) on wasm.
#[cfg(not(target_arch = "wasm32"))]
pub fn decode_child(_node_index: usize) -> VortexResult<Vec<u8>> {
    vortex_bail!("decode_child is only available inside a running wasm kernel")
}

/// Emit a debug log line to the host.
#[cfg(target_arch = "wasm32")]
pub fn log(message: &str) {
    unsafe { imports::vx_host_log(message.as_ptr() as i32, message.len() as i32) };
}

/// No-op host log stub for non-wasm targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn log(_message: &str) {}

/// Allocate `len` bytes in the guest's linear memory and return the pointer.
///
/// The allocation is intentionally leaked: the host reads any returned data before the kernel's
/// store (and thus its entire linear memory) is dropped after the decode call completes.
#[doc(hidden)]
pub fn __alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}
