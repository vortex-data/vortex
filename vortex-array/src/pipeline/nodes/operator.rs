// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operators are physical execution nodes of a pipeline.

use crate::pipeline::PipelineContext;
use crate::pipeline::bits::BitView;
use crate::pipeline::vector::Vector;
use std::cell::Ref;
use std::task::Poll;
use vortex_error::VortexResult;

/// Execution phase operator - does the actual computation
pub trait Operator: 'static {
    /// Get metadata about this operator
    /// fn metadata(&self) -> OperatorMetadata;

    /// Execute with dynamic dispatch (fallback)
    fn execute_dyn(
        &mut self,
        ctx: &dyn PipelineContext,
        mask: BitView,
        inputs: &[Ref<Vector>],
        output: &mut Vector,
    ) -> Poll<VortexResult<()>>;
}
