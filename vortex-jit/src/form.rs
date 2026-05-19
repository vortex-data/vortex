// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::types as cl_types;
use cranelift::prelude::Type as ClType;

/// Primitive element type carried through the pipeline.
///
/// v0 supports a subset; expansion is trivial — match arms in `cl_type` and
/// `byte_width`, plus matching `IntBuilder` ops in `emit.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PType {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
}

impl PType {
    pub const fn cl_type(self) -> ClType {
        match self {
            Self::I32 | Self::U32 => cl_types::I32,
            Self::I64 | Self::U64 => cl_types::I64,
            Self::F32 => cl_types::F32,
            Self::F64 => cl_types::F64,
        }
    }

    /// 128-bit (SSE2) SIMD type for this primitive.
    ///
    /// Cranelift's x86_64 backend currently only supports 128-bit codegen
    /// (verified: i32x8 emit fails with "Unsupported feature: Unexpected
    /// SSA-value type: i32x8" even on AVX2/AVX-512 hosts).
    ///
    /// LLVM uses 256-bit (AVX2) or 512-bit (AVX-512) where available, so the
    /// JIT will trail LLVM autovec on bandwidth-bound elementwise ops by the
    /// vector-width ratio. The framework will get wider SIMD automatically
    /// once Cranelift's backend supports it — no API changes needed.
    pub const fn simd_type(self) -> ClType {
        match self {
            Self::I32 | Self::U32 => cl_types::I32X4,
            Self::I64 | Self::U64 => cl_types::I64X2,
            Self::F32 => cl_types::F32X4,
            Self::F64 => cl_types::F64X2,
        }
    }

    /// Number of scalar lanes per SIMD chunk (128-bit / sizeof(T)).
    pub const fn simd_lanes(self) -> u32 {
        match self {
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 2,
        }
    }

    pub const fn byte_width(self) -> u32 {
        match self {
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    pub const fn is_int(self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::U32 | Self::U64)
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }
}

/// Memory layout of a lane stream.
///
/// `Linear` is the natural per-element order. `FastLanesTransposed(N)` indicates
/// the fastlanes per-lane interleaved layout (16 lanes for u64, 32 for u32, ...);
/// stages that consume this need an `Untranspose` step before any linear-order
/// consumer like ALP. v0 only exercises `Linear`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    Linear,
    FastLanesTransposed(u8),
    /// "Whatever the producer says" — used by stages that don't care.
    Either,
}

/// The form of data flowing between stages.
///
/// A stage's `input()` and `output()` are declared as `Form`. The framework's
/// `Pipeline::push` matches adjacent stages on form and rejects incompatible
/// pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Form {
    /// SSA-resident values, ready for elementwise ops.
    Lane(PType, Layout),
    /// Buffer in scratch memory; needed for cross-element communication.
    Buffer(PType, Layout),
    /// Pipeline leaf (no input) or terminal (no output).
    None,
}

impl Form {
    pub(crate) fn compatible(self, next_input: Self) -> bool {
        use Form::{Buffer, Lane, None};
        match (self, next_input) {
            (None, None) => true,
            (Lane(t, l), Lane(t2, l2)) | (Buffer(t, l), Buffer(t2, l2)) => {
                t == t2 && layout_compatible(l, l2)
            }
            _ => false,
        }
    }

    pub(crate) fn ptype(self) -> Option<PType> {
        match self {
            Self::Lane(t, _) | Self::Buffer(t, _) => Some(t),
            Self::None => None,
        }
    }
}

fn layout_compatible(a: Layout, b: Layout) -> bool {
    matches!((a, b), (_, Layout::Either))
        || matches!((a, b), (Layout::Either, _))
        || a == b
}
