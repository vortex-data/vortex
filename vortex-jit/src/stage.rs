// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use crate::emit::{EmitCtx, SigBuilder};
use crate::form::Form;

/// An emittable pipeline stage.
///
/// Stages own three responsibilities:
///   - declare what shape of data they expect / produce (`input`, `output`),
///   - declare what runtime args and externs they need (`declare`),
///   - emit IR for one block (`emit`).
///
/// The framework owns the outer block loop, the cross-stage lane handoff,
/// and the post-loop tail (for stages that emit a `BlockKind::PostLoop` body).
pub trait JitStage: Send + Sync + std::fmt::Debug {
    /// Stable name for fingerprinting/debug.
    fn tag(&self) -> &'static str;

    /// Const params hashed into kernel identity (bit_width, reference, ...).
    /// Two stages with identical `tag()` and `fingerprint()` produce identical IR.
    fn fingerprint(&self) -> Vec<u8> {
        Vec::new()
    }

    /// One entry per input slot. Empty for leaf stages.
    fn input(&self) -> SmallVec<[Form; 1]> {
        SmallVec::new()
    }

    /// Output form. `Form::None` for terminals.
    fn output(&self) -> Form {
        Form::None
    }

    /// When this stage emits its IR.
    fn placement(&self) -> Placement {
        Placement::InBlock
    }

    /// Request runtime args / externs at compile time. Called once.
    fn declare(&self, _sig: &mut SigBuilder) {}

    /// Emit IR. Called once per block for `InBlock`, once total for `PostLoop`.
    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()>;
}

/// Where a stage's IR lives relative to the block loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Inside the block loop body. Most stages.
    InBlock,
    /// After the block loop completes. For patch-application style stages
    /// that operate on the fully-assembled output buffer.
    PostLoop,
}
