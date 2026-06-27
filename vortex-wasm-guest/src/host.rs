// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest allocator and access to host imports (the `vortex_host` module).

use crate::arrow::ChildView;
use crate::error::GuestError;
use crate::error::GuestResult;

#[cfg(target_arch = "wasm32")]
mod imports {
    #[link(wasm_import_module = "vortex_host")]
    unsafe extern "C" {
        pub fn vx_decode_child(node_index: i32, out_ptr: i32) -> i32;
    }
}

/// Ask the host to decode the child array at `node_index` and return a view of it.
///
/// The host writes the child as Arrow C Data Interface structs into this module's memory and
/// stores the `(array_ptr, schema_ptr)` pair at a scratch location we pass in.
#[cfg(target_arch = "wasm32")]
pub fn decode_child(node_index: usize) -> GuestResult<ChildView> {
    let mut out = [0u8; 8];
    let rc = unsafe { imports::vx_decode_child(node_index as i32, out.as_mut_ptr() as i32) };
    if rc != 0 {
        return Err(GuestError::new("host vx_decode_child failed"));
    }
    let array_ptr = u32::from_le_bytes(out[0..4].try_into().expect("4 bytes"));
    let schema_ptr = u32::from_le_bytes(out[4..8].try_into().expect("4 bytes"));
    crate::arrow::read_child(array_ptr, schema_ptr)
}

/// Host stub for non-wasm targets (so the SDK builds on the host for unit tests).
#[cfg(not(target_arch = "wasm32"))]
pub fn decode_child(_node_index: usize) -> GuestResult<ChildView> {
    Err(GuestError::new(
        "decode_child is only available inside a running wasm kernel",
    ))
}

/// Allocate `len` bytes in linear memory and return the offset.
///
/// The allocation is leaked: the host reads any returned data before the kernel's store (and thus
/// its entire linear memory) is dropped after the decode call.
#[doc(hidden)]
pub fn __alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Allocate `bytes.len()` bytes, copy `bytes` in, and return the offset.
pub(crate) fn alloc_bytes(bytes: &[u8]) -> u32 {
    let ptr = __alloc(bytes.len().max(1));
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    ptr as u32
}
