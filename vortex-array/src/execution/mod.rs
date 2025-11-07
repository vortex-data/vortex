// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod batch;
mod mask;
mod validity;

pub use batch::*;
pub use mask::*;

/// Execution context for batch array compute.
// NOTE(ngates): This context will eventually hold cached resources for execution, such as CSE
//  nodes, and may well eventually support a type-map interface for arrays to stash arbitrary
//  execution-related data.
pub trait ExecutionCtx: private::Sealed {}

/// A crate-internal dummy execution context.
pub(crate) struct DummyExecutionCtx;
impl ExecutionCtx for DummyExecutionCtx {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::DummyExecutionCtx {}
}
