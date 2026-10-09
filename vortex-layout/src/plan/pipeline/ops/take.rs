// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Looking codes up in dictionary values.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Shared;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
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
///
/// The values usually arrive as one shared array, read once for the whole scan; values read
/// for this take alone are joined and shared here.
pub(crate) struct TakeSource {
    plan: TakePlan,
    values: Option<ArrayRef>,
}

const CODES: usize = 0;
const VALUES: usize = 1;

impl TakeSource {
    /// A take of its codes inlet's batches over its values inlet's.
    pub(crate) fn new(plan: TakePlan) -> Self {
        Self { plan, values: None }
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
                let values = shared(join(self.plan.values()?.dtype(), drain(&mut inlet))?);
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
        2
    }

    fn capacity(&self, inlet: usize) -> usize {
        if inlet == VALUES {
            UNBOUNDED
        } else {
            DEFAULT_CAPACITY
        }
    }
}

/// `array` as a [`SharedArray`], so however many dictionaries use it, it is canonicalized once.
fn shared(array: ArrayRef) -> ArrayRef {
    if array.is::<Shared>() {
        array
    } else {
        SharedArray::new(array).into_array()
    }
}

/// Joins every batch of a plan's whole output into one shared array, emitted at the end.
pub(crate) struct WholeStage {
    dtype: DType,
    batches: Vec<ArrayRef>,
}

impl WholeStage {
    pub(crate) fn new(dtype: DType) -> Self {
        Self {
            dtype,
            batches: Vec::new(),
        }
    }
}

impl Operator for WholeStage {
    fn compute(&mut self, input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => {
                self.batches.push(batch);
                Ok(Step::Consumed)
            }
            Input::End => {
                let batches = std::mem::take(&mut self.batches);
                Ok(Step::Last(shared(join(&self.dtype, batches)?)))
            }
            Input::None => Ok(Step::Consumed),
        }
    }
}
