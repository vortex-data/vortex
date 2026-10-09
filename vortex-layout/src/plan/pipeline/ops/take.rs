// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Looking codes up in dictionary values.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Shared;
use vortex_array::arrays::SharedArray;
use vortex_error::VortexResult;

use super::*;
use crate::plan::TakePlan;
use crate::plan::pipeline::Blocked;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::DEFAULT_CAPACITY;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;

/// Wraps each batch of codes as a dictionary over the values, once the values inlet has ended.
pub(crate) struct TakeSource {
    plan: TakePlan,
    values: Option<ArrayRef>,
}

const CODES: usize = 0;
const VALUES: usize = 1;

impl TakeSource {
    /// A take over `values`, or over what its values inlet produces when `None`.
    pub(crate) fn new(plan: TakePlan, values: Option<ArrayRef>) -> Self {
        Self { plan, values }
    }
}

impl Operator for TakeSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let values = match &self.values {
            Some(values) => values.clone(),
            None => {
                let mut inlet = cx.inlet(VALUES);
                if !inlet.closed() {
                    return Ok(Step::Blocked(Blocked::Inlet(VALUES)));
                }
                let values = join(self.plan.values()?.dtype(), drain(&mut inlet))?;
                let values = if values.is::<Shared>() {
                    values
                } else {
                    SharedArray::new(values).into_array()
                };
                let values = self.plan.cache_values(values);
                self.values = Some(values.clone());
                values
            }
        };
        let mut codes = cx.inlet(CODES);
        match codes.take() {
            Some(batch) => {
                let more = !codes.is_empty();
                let array = DictArray::try_new(batch, values)?.into_array();
                Ok(if more {
                    Step::More(array)
                } else {
                    Step::Last(array)
                })
            }
            None if codes.closed() => Ok(Step::Finished),
            None => Ok(Step::Blocked(Blocked::Inlet(CODES))),
        }
    }
}

impl Source for TakeSource {
    fn inlet_count(&self) -> usize {
        if self.values.is_some() { 1 } else { 2 }
    }

    fn capacity(&self, inlet: usize) -> usize {
        if inlet == VALUES {
            UNBOUNDED
        } else {
            DEFAULT_CAPACITY
        }
    }
}
