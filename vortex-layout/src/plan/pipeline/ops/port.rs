// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sources that read one port, or emit one batch.

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::plan::pipeline::Blocked;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;

/// Emits its inlet's batches as they arrive.
pub(crate) struct PortSource;

impl Operator for PortSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let mut inlet = cx.inlet(0);
        match inlet.take() {
            Some(batch) if inlet.is_empty() => Ok(Step::Last(batch)),
            Some(batch) => Ok(Step::More(batch)),
            None if inlet.closed() => Ok(Step::Finished),
            None => Ok(Step::Blocked(Blocked::Inlet(0))),
        }
    }
}

impl Source for PortSource {
    fn inlet_count(&self) -> usize {
        1
    }
}

/// Emits one batch it was built with.
pub(crate) struct OnceSource(Option<ArrayRef>);

impl OnceSource {
    pub(crate) fn new(batch: ArrayRef) -> Self {
        Self(Some(batch))
    }
}

impl Operator for OnceSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match self.0.take() {
            Some(batch) => Step::Last(batch),
            None => Step::Finished,
        })
    }
}

impl Source for OnceSource {}
