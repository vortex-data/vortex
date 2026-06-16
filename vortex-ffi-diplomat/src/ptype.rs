// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex primitive types (`PType`).
//!
//! In the hand-written C ABI this was a `#[repr(C)]` C-like enum named `vx_ptype` with
//! explicit discriminants (`PTYPE_U8 = 0`, ... `PTYPE_F64 = 10`) plus two `From`
//! conversions to and from `vortex::dtype::PType`. Under Diplomat the same set of variants
//! becomes a plain Diplomat enum, `VxPType`, which Diplomat renders as a native enum in each
//! target language. The conversion helpers are kept as ordinary (non-`#[diplomat::bridge]`)
//! Rust so that the sibling `dtype`/`scalar` bridges can translate between `VxPType` and the
//! core `PType`.

pub use ffi::VxPType;

#[diplomat::bridge]
pub mod ffi {
    /// A Vortex primitive type.
    ///
    /// Mirrors `vortex::dtype::PType`. This replaces the C `vx_ptype` enum; the variant set is
    /// identical, but Diplomat enums use idiomatic names rather than the `PTYPE_`-prefixed C
    /// constants.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum VxPType {
        /// Unsigned 8-bit integer.
        U8,
        /// Unsigned 16-bit integer.
        U16,
        /// Unsigned 32-bit integer.
        U32,
        /// Unsigned 64-bit integer.
        U64,
        /// Signed 8-bit integer.
        I8,
        /// Signed 16-bit integer.
        I16,
        /// Signed 32-bit integer.
        I32,
        /// Signed 64-bit integer.
        I64,
        /// 16-bit floating point number.
        F16,
        /// 32-bit floating point number.
        F32,
        /// 64-bit floating point number.
        F64,
    }
}

impl From<VxPType> for vortex::dtype::PType {
    fn from(value: VxPType) -> Self {
        use vortex::dtype::PType;
        match value {
            VxPType::U8 => PType::U8,
            VxPType::U16 => PType::U16,
            VxPType::U32 => PType::U32,
            VxPType::U64 => PType::U64,
            VxPType::I8 => PType::I8,
            VxPType::I16 => PType::I16,
            VxPType::I32 => PType::I32,
            VxPType::I64 => PType::I64,
            VxPType::F16 => PType::F16,
            VxPType::F32 => PType::F32,
            VxPType::F64 => PType::F64,
        }
    }
}

impl From<vortex::dtype::PType> for VxPType {
    fn from(value: vortex::dtype::PType) -> Self {
        use vortex::dtype::PType;
        match value {
            PType::U8 => VxPType::U8,
            PType::U16 => VxPType::U16,
            PType::U32 => VxPType::U32,
            PType::U64 => VxPType::U64,
            PType::I8 => VxPType::I8,
            PType::I16 => VxPType::I16,
            PType::I32 => VxPType::I32,
            PType::I64 => VxPType::I64,
            PType::F16 => VxPType::F16,
            PType::F32 => VxPType::F32,
            PType::F64 => VxPType::F64,
        }
    }
}
