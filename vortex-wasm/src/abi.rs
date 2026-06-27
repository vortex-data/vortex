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

/// Guest export: `vx_decode(input_ptr: i32, input_len: i32) -> i32`. Decodes the serialized array
/// and returns the offset of a length-prefixed [`CanonicalMessage`](crate::message), i.e.
/// `[u32 len][message bytes…]`.
pub const DECODE_EXPORT: &str = "vx_decode";

/// Host import: `vx_decode_child(node_index: i32, out_ptr: i32) -> i32`. The host decodes the
/// child array at `node_index`, writes a [`CanonicalMessage`](crate::message) into freshly
/// allocated guest memory, and stores the `(offset: u32, len: u32)` pair at `out_ptr`.
pub const DECODE_CHILD_IMPORT: &str = "vx_decode_child";

/// Host import: `vx_host_log(ptr: i32, len: i32)`. Logs a UTF-8 string from guest memory.
pub const HOST_LOG_IMPORT: &str = "vx_host_log";

/// Discriminant for the canonical array kind in a [`CanonicalMessage`](crate::message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageKind {
    /// A `NullArray` — only a length.
    Null = 0,
    /// A `BoolArray`.
    Bool = 1,
    /// A `PrimitiveArray`.
    Primitive = 2,
    /// A `VarBinViewArray` (utf8 / binary).
    VarBinView = 3,
    /// A `StructArray`.
    Struct = 4,
}

impl MessageKind {
    /// Convert from the on-wire discriminant.
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Null,
            1 => Self::Bool,
            2 => Self::Primitive,
            3 => Self::VarBinView,
            4 => Self::Struct,
            _ => return None,
        })
    }
}

/// Discriminant for the validity representation in a [`CanonicalMessage`](crate::message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageValidity {
    /// The dtype is non-nullable.
    NonNullable = 0,
    /// Nullable, but every element is valid.
    AllValid = 1,
    /// Nullable, every element is null.
    AllInvalid = 2,
    /// A validity bitmap stored as buffer index 1 (buffer 0 holds the values).
    Bitmap = 3,
}

impl MessageValidity {
    /// Convert from the on-wire discriminant.
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::NonNullable,
            1 => Self::AllValid,
            2 => Self::AllInvalid,
            3 => Self::Bitmap,
            _ => return None,
        })
    }
}

/// Size in bytes of the fixed [`CanonicalMessage`](crate::message) header preceding the buffer
/// table.
///
/// Layout: `u8 kind | u8 ptype | u8 validity | u8 pad | u64 length | u32 nbuffers | u32 nchildren`.
pub const MESSAGE_HEADER_LEN: usize = 20;

/// Size in bytes of each buffer-table entry header preceding its inline bytes.
///
/// Layout: `u64 len | u8 alignment_exponent | u8[7] pad`.
pub const BUFFER_ENTRY_HEADER_LEN: usize = 16;
