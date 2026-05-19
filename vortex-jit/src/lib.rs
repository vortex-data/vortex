// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cranelift-backed JIT compilation framework for fused decompression pipelines.
//!
//! See `design-notes/jit-fusion-prototype.md` for the full design discussion.
//! This crate implements Approach A: stages exchange `LaneSlice<T>` (a `Vec` of
//! SSA Values held during codegen) via `EmitCtx::take_input` / `put_output`,
//! and the framework chains them in one Cranelift function so the SSA Values
//! flow stage-to-stage without ever materializing to memory.
//!
//! v0 limitations:
//!   - Scalar lanes only (no `i32x8` IR emission yet; the `Form::Lane` carries
//!     `Layout::Linear` and stages operate per-element).
//!   - No bit-packed leaf — input is a plain primitive buffer.
//!   - Two stages implemented: `ForAdd`, `Delta`. Plus terminal `StoreOut` and
//!     a post-loop `ApplyPatches`.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

mod compiler;
mod emit;
mod form;
mod pipeline;
mod stage;
pub mod stages;

pub use compiler::{Compiled, Compiler, ExternFn, KernelOp};
pub use emit::{ArgKey, EmitCtx, ExternId, LaneSlice, Lanes, Scalar, SigBuilder};
pub use form::{Form, Layout, PType};
pub use pipeline::{DecodeNode, Pipeline};
pub use stage::JitStage;
