// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concatenating inlets in order.

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;

/// Emits its inlets one after another, each to its end.
pub(crate) struct ConcatSource {
    inlets: usize,
    /// The inlet being emitted; those before it have ended.
    current: usize,
}

impl ConcatSource {
    pub(crate) fn new(inlets: usize) -> Self {
        Self { inlets, current: 0 }
    }
}

impl Source for ConcatSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        while self.current < self.inlets {
            let inlet = cx.inlet(self.current);
            if !inlet.is_empty() {
                return Ok(SourceReady::Ready);
            }
            if !inlet.closed() {
                return Ok(SourceReady::Blocked(SourceBlocked::Inlet(self.current)));
            }
            self.current += 1;
        }
        Ok(SourceReady::Ready)
    }
}

impl Operator for ConcatSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(if self.current == self.inlets {
            Step::End
        } else {
            Step::Final(cx.inlet(self.current).take().ok_or_else(
                || vortex_err!(AssertionFailed: "Concat computed before its inlet was ready"),
            )?)
        })
    }
}
