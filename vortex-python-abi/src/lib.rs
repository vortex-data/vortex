// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared private ABI for Python buffer handoff between `vortex-data` extension modules.

use std::ffi::CStr;
use std::ffi::c_void;

/// Name used for PyCapsules carrying [`VortexBufferExport`] pointers.
pub const BUFFER_EXPORT_CAPSULE_NAME: &CStr = c"vortex_buffer_export";

/// Current version of the [`VortexBufferExport`] ABI.
pub const VORTEX_BUFFER_EXPORT_VERSION: u32 = 1;

/// Buffer kind for host-accessible buffers.
pub const VORTEX_BUFFER_HOST: u32 = 0;

/// Buffer kind for device-accessible buffers.
pub const VORTEX_BUFFER_DEVICE: u32 = 1;

/// C-ABI descriptor for passing buffers between `vortex-data` and optional extension modules.
///
/// This type is shared by Rust crates, but the values are exchanged through Python capsules between
/// independently compiled extension modules. The producer owns allocation details and must provide a
/// `release` callback that releases both `private_data` and the descriptor itself.
#[repr(C)]
pub struct VortexBufferExport {
    /// ABI version. Consumers must reject unsupported versions.
    pub version: u32,
    /// Buffer kind. Consumers may support [`VORTEX_BUFFER_HOST`] or [`VORTEX_BUFFER_DEVICE`].
    pub kind: u32,
    /// Pointer to the first byte of the exported buffer, or null for empty buffers.
    pub ptr: *const u8,
    /// Length of the buffer in bytes.
    pub len: usize,
    /// Required byte alignment of `ptr`.
    pub alignment: usize,
    /// Device identifier for device buffers, or -1 for host buffers.
    pub device_id: i32,
    /// Optional synchronization event for device buffers.
    pub sync_event: *mut c_void,
    /// Producer-owned private data used by `release`.
    pub private_data: *mut c_void,
    /// Producer-owned release callback. It must release `private_data` and this descriptor.
    pub release: Option<unsafe extern "C" fn(*mut VortexBufferExport)>,
}
