// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The host/guest ABI shared between [`crate::WasmKernel`] and the `vortex-wasm-guest` SDK.
//!
//! All multi-byte integers in the wire format are little-endian. See
//! `docs/design/wasm-encodings.md` for the full specification.

/// Version of the host/guest ABI implemented by this crate.
///
/// A kernel records the ABI version it was built against in its [`WasmLayoutMetadata`]; the host
/// refuses to run a kernel whose ABI version it does not understand.
///
/// [`WasmLayoutMetadata`]: crate::WasmLayoutMetadata
pub const ABI_VERSION: u32 = 1;

/// Name of the host import module that the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// Name of the guest's exported linear memory.
pub const MEMORY_EXPORT: &str = "memory";

/// Guest export: `vx_alloc(len: i32) -> i32`. Allocates `len` bytes, returns the offset.
pub const ALLOC_EXPORT: &str = "vx_alloc";

/// Guest export: `vx_decode(input_ptr: i32, input_len: i32) -> i32`. Decodes the encoding-specific
/// input and returns the offset of an `(array_ptr: u32, schema_ptr: u32)` pair pointing at the
/// decoded array's Arrow C Data Interface structs (see [`crate::arrow_ffi`]).
pub const DECODE_EXPORT: &str = "vx_decode";

/// Host import: `vx_decode_child(node_index: i32, out_ptr: i32) -> i32`. The host decodes the
/// child array at `node_index`, writes its Arrow C Data Interface structs into freshly allocated
/// guest memory, and stores the `(array_ptr: u32, schema_ptr: u32)` pair at `out_ptr`.
pub const DECODE_CHILD_IMPORT: &str = "vx_decode_child";

/// Host import: `vx_host_log(ptr: i32, len: i32)`. Logs a UTF-8 string from guest memory.
pub const HOST_LOG_IMPORT: &str = "vx_host_log";
