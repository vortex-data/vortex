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
    /// Bit width used to pack the encoded i32 codes (0..=32). Selects which
    /// `cp_unpack_u32_bw<N>` PTX module the executor links — the constant
    /// is baked into the stencil at compile time rather than flowing as a
    /// kernel argument.
    pub bit_width: u8,
    /// ALP factor `F10[exponents.f]`.
    pub f: f32,
    /// ALP factor `IF10[exponents.e]`.
    pub e: f32,
    /// Tail of the pipeline.
    pub post: PostOp,
}

/// Per-bit-width unpack-stencil module names. Generated at build time by
/// `vortex-cuda/build.rs::generate_cp_unpack_u32_stencils`. Every entry
/// exports the same symbol `cp_unpack`; selection happens at link time.
const CP_UNPACK_U32_MODULES: [&str; 33] = [
    "cp_unpack_u32_bw0",
    "cp_unpack_u32_bw1",
    "cp_unpack_u32_bw2",
    "cp_unpack_u32_bw3",
    "cp_unpack_u32_bw4",
    "cp_unpack_u32_bw5",
    "cp_unpack_u32_bw6",
    "cp_unpack_u32_bw7",
    "cp_unpack_u32_bw8",
    "cp_unpack_u32_bw9",
    "cp_unpack_u32_bw10",
    "cp_unpack_u32_bw11",
    "cp_unpack_u32_bw12",
    "cp_unpack_u32_bw13",
    "cp_unpack_u32_bw14",
    "cp_unpack_u32_bw15",
    "cp_unpack_u32_bw16",
    "cp_unpack_u32_bw17",
    "cp_unpack_u32_bw18",
    "cp_unpack_u32_bw19",
    "cp_unpack_u32_bw20",
    "cp_unpack_u32_bw21",
    "cp_unpack_u32_bw22",
    "cp_unpack_u32_bw23",
    "cp_unpack_u32_bw24",
    "cp_unpack_u32_bw25",
    "cp_unpack_u32_bw26",
    "cp_unpack_u32_bw27",
    "cp_unpack_u32_bw28",
    "cp_unpack_u32_bw29",
    "cp_unpack_u32_bw30",
    "cp_unpack_u32_bw31",
    "cp_unpack_u32_bw32",
];

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

    /// Unpack stencil module for this plan's bit width. Panics for widths
    /// outside `0..=32`; the executor pre-validates so this is a programmer
    /// error.
    pub fn unpack_module(&self) -> &'static str {
        CP_UNPACK_U32_MODULES[self.bit_width as usize]
    }

    /// PTX modules that must be linked to satisfy the trampoline's `extern`
    /// references, in arbitrary order.
    pub fn stencil_modules(&self) -> [&'static str; 3] {
        let post = match self.post {
            PostOp::Arith { op, .. } => op.stencil_module(),
            PostOp::Filter { op, .. } => op.stencil_module(),
        };
        [self.unpack_module(), "cp_alp_apply_i32_f32", post]
    }
}
