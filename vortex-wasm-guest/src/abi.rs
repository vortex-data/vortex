// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host/guest ABI constants and the Arrow C Data Interface layout (wasm32) shared with the host.
//!
//! Decoded arrays cross the boundary as the [Arrow C Data Interface]: the guest builds and reads
//! the `ArrowSchema`/`ArrowArray` structs directly (no nanoarrow needed — they are plain byte
//! layouts). These offsets MUST match `vortex-wasm`'s `arrow_ffi` module.
//!
//! [Arrow C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface.html

/// Host/guest ABI version.
pub const ABI_VERSION: u32 = 1;

/// Host import module name the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// Size of an `ArrowSchema` struct in the wasm32 C ABI.
pub const SCHEMA_SIZE: usize = 48;
/// Size of an `ArrowArray` struct in the wasm32 C ABI.
pub const ARRAY_SIZE: usize = 64;

/// `ArrowSchema` field offsets (wasm32 C ABI: 4-byte pointers, 8-aligned `int64`).
pub mod schema {
    /// `const char* format`
    pub const FORMAT: usize = 0;
    /// `int64 flags`
    pub const FLAGS: usize = 16;
}

/// `ArrowArray` field offsets (wasm32 C ABI).
pub mod array {
    /// `int64 length`
    pub const LENGTH: usize = 0;
    /// `int64 null_count`
    pub const NULL_COUNT: usize = 8;
    /// `int64 offset`
    pub const OFFSET: usize = 16;
    /// `int64 n_buffers`
    pub const N_BUFFERS: usize = 24;
    /// `const void** buffers`
    pub const BUFFERS: usize = 40;
}

/// Arrow schema flag: the field may contain nulls.
pub const ARROW_FLAG_NULLABLE: i64 = 2;

/// Primitive type, matching Vortex `PType` and the Arrow C Data Interface format codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PType {
    /// `u8`
    U8,
    /// `u16`
    U16,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `i8`
    I8,
    /// `i16`
    I16,
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `f16`
    F16,
    /// `f32`
    F32,
    /// `f64`
    F64,
}

impl PType {
    /// Width in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            PType::U8 | PType::I8 => 1,
            PType::U16 | PType::I16 | PType::F16 => 2,
            PType::U32 | PType::I32 | PType::F32 => 4,
            PType::U64 | PType::I64 | PType::F64 => 8,
        }
    }

    /// Arrow C Data Interface format code (no trailing NUL).
    pub const fn format_code(self) -> &'static str {
        match self {
            PType::I8 => "c",
            PType::U8 => "C",
            PType::I16 => "s",
            PType::U16 => "S",
            PType::I32 => "i",
            PType::U32 => "I",
            PType::I64 => "l",
            PType::U64 => "L",
            PType::F16 => "e",
            PType::F32 => "f",
            PType::F64 => "g",
        }
    }

    /// Parse an Arrow C Data Interface primitive format code.
    pub fn from_format(format: &str) -> Option<Self> {
        Some(match format {
            "c" => PType::I8,
            "C" => PType::U8,
            "s" => PType::I16,
            "S" => PType::U16,
            "i" => PType::I32,
            "I" => PType::U32,
            "l" => PType::I64,
            "L" => PType::U64,
            "e" => PType::F16,
            "f" => PType::F32,
            "g" => PType::F64,
            _ => return None,
        })
    }
}
