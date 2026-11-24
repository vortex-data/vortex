// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Collecting an inlet whole before producing one batch from it.

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use super::join_inlet;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;

type Finish = Box<dyn FnOnce(ArrayRef, &mut Cx<'_>) -> VortexResult<ArrayRef> + Send>;

/// Joins the whole of its one inlet, of `dtype`, once it has closed, and emits what `finish`
/// makes of it. Its inlet should be unbounded.
pub(crate) struct WholeSource {
    dtype: DType,
    finish: Option<Finish>,
}

impl WholeSource {
    pub(crate) fn new(
        dtype: DType,
        finish: impl FnOnce(ArrayRef, &mut Cx<'_>) -> VortexResult<ArrayRef> + Send + 'static,
    ) -> Self {
        Self {
            dtype,
            finish: Some(Box::new(finish)),
        }
    }
}

impl Source for WholeSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        Ok(if self.finish.is_some() && !cx.inlet(0).closed() {
            SourceReady::Blocked(SourceBlocked::Inlet(0))
        } else {
            SourceReady::Ready
        })
    }
}

impl Operator for WholeSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Some(finish) = self.finish.take() else {
            return Ok(Step::End);
        };
        let whole = join_inlet(cx, 0, &self.dtype)?;
        Ok(Step::Final(finish(whole, cx)?))
    }
}
