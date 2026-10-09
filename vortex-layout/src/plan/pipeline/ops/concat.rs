// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concatenating inlets in order.

use vortex_error::VortexResult;

use crate::plan::pipeline::Blocked;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;

/// Emits its inlets one after another, each to its end.
pub(crate) struct ConcatSource {
    inlets: usize,
    current: usize,
}

impl ConcatSource {
    pub(crate) fn new(inlets: usize) -> Self {
        Self { inlets, current: 0 }
    }
}

impl Operator for ConcatSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        while self.current < self.inlets {
            let mut inlet = cx.inlet(self.current);
            if let Some(batch) = inlet.take() {
                return Ok(if inlet.is_empty() && !inlet.closed() {
                    Step::Last(batch)
                } else {
                    Step::More(batch)
                });
            }
            if !inlet.closed() {
                return Ok(Step::Blocked(Blocked::Inlet(self.current)));
            }
            self.current += 1;
        }
        Ok(Step::Finished)
    }
}

impl Source for ConcatSource {
    fn inlet_count(&self) -> usize {
        self.inlets
    }
}
