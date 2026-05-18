// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan IR for the Copy-and-Patch demo.
//!
//! A `Plan` describes a fused decode + post-op pipeline whose stencil
//! variants are picked at runtime. The current scope is the (u32 bitpacked
//! → i32 ALP → f32) decode flavour with an arithmetic or filter tail;
//! other widths plug in by adding their PTX stencils and a matching
//! trampoline.

/// Element-wise arithmetic op against a scalar constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    /// `out[i] = x[i] + c`
    Add,
    /// `out[i] = x[i] * c`
    Mul,
}

impl ArithOp {
    /// Stencil PTX module name (matches the `.cu` basename in `kernels/src/copy_patch/`).
    pub fn stencil_module(self) -> &'static str {
        match self {
            Self::Add => "cp_arith_add_f32",
            Self::Mul => "cp_arith_mul_f32",
        }
    }
}

/// Element-wise comparison op against a scalar constant. Produces a `u8`
/// mask (0/1 per element) in the prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// `mask[i] = x[i] > c`
    Gt,
    /// `mask[i] = x[i] < c`
    Lt,
    /// `mask[i] = x[i] == c`
    Eq,
}

impl FilterOp {
    pub fn stencil_module(self) -> &'static str {
        match self {
            Self::Gt => "cp_filter_gt_f32",
            Self::Lt => "cp_filter_lt_f32",
            Self::Eq => "cp_filter_eq_f32",
        }
    }
}

/// Terminal stage of the pipeline. Determines which trampoline is linked
/// (`arith` vs `filter`) and which post-op stencil is linked into it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PostOp {
    Arith { op: ArithOp, c: f32 },
    Filter { op: FilterOp, c: f32 },
}

/// A Copy-and-Patch query plan over u32 bitpacked + ALP-encoded data.
///
/// Fields are kept flat because the prototype handles a single decode shape;
/// adding more is a matter of new trampoline variants and a wider enum here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Bit width used to pack the encoded i32 codes. Passed as a kernel param
    /// to the prototype unpack stencil (per-bw specialization is a follow-up).
    pub bit_width: u8,
    /// ALP factor `F10[exponents.f]`.
    pub f: f32,
    /// ALP factor `IF10[exponents.e]`.
    pub e: f32,
    /// Tail of the pipeline.
    pub post: PostOp,
}

impl Plan {
    /// The trampoline PTX module name that this plan dispatches through.
    pub fn trampoline_module(&self) -> &'static str {
        match self.post {
            PostOp::Arith { .. } => "cp_trampoline_u32_f32_arith",
            PostOp::Filter { .. } => "cp_trampoline_u32_f32_filter",
        }
    }

    /// The CUDA kernel function name exported by the trampoline module.
    pub fn trampoline_entry(&self) -> &'static str {
        match self.post {
            PostOp::Arith { .. } => "cp_trampoline_arith_u32_f32",
            PostOp::Filter { .. } => "cp_trampoline_filter_u32_f32",
        }
    }

    /// PTX modules that must be linked to satisfy the trampoline's `extern`
    /// references, in arbitrary order.
    pub fn stencil_modules(&self) -> [&'static str; 3] {
        let post = match self.post {
            PostOp::Arith { op, .. } => op.stencil_module(),
            PostOp::Filter { op, .. } => op.stencil_module(),
        };
        ["cp_unpack_u32", "cp_alp_apply_i32_f32", post]
    }
}
