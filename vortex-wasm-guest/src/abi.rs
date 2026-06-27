// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest-side mirror of the host/guest ABI constants.
//!
//! These MUST stay byte-compatible with `vortex-wasm`'s `abi` module. See
//! `docs/design/wasm-encodings.md`.

/// Version of the host/guest ABI implemented by this SDK.
pub const ABI_VERSION: u32 = 1;

/// Name of the host import module the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// `kind` discriminant for a [`CanonicalMessage`](crate::message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageKind {
    /// A `NullArray`.
    Null = 0,
    /// A `BoolArray`.
    Bool = 1,
    /// A `PrimitiveArray`.
    Primitive = 2,
    /// A `VarBinViewArray`.
    VarBinView = 3,
    /// A `StructArray`.
    Struct = 4,
}

/// `validity` discriminant for a [`CanonicalMessage`](crate::message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageValidity {
    /// Non-nullable dtype.
    NonNullable = 0,
    /// Nullable, all valid.
    AllValid = 1,
    /// Nullable, all null.
    AllInvalid = 2,
    /// Validity bitmap stored as buffer index 1 (buffer 0 holds the values).
    Bitmap = 3,
}

/// Primitive type discriminant matching Vortex `PType` and the array flatbuffer schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PType {
    /// `u8`
    U8 = 0,
    /// `u16`
    U16 = 1,
    /// `u32`
    U32 = 2,
    /// `u64`
    U64 = 3,
    /// `i8`
    I8 = 4,
    /// `i16`
    I16 = 5,
    /// `i32`
    I32 = 6,
    /// `i64`
    I64 = 7,
    /// `f16`
    F16 = 8,
    /// `f32`
    F32 = 9,
    /// `f64`
    F64 = 10,
}

impl PType {
    /// Width of this primitive type in bytes.
    pub fn byte_width(self) -> usize {
        match self {
            PType::U8 | PType::I8 => 1,
            PType::U16 | PType::I16 | PType::F16 => 2,
            PType::U32 | PType::I32 | PType::F32 => 4,
            PType::U64 | PType::I64 | PType::F64 => 8,
        }
    }
}

/// Size of the fixed [`CanonicalMessage`](crate::message) header.
pub const MESSAGE_HEADER_LEN: usize = 20;

/// Size of each buffer-table entry header.
pub const BUFFER_ENTRY_HEADER_LEN: usize = 16;
