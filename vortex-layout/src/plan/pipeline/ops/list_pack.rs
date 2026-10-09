// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from offsets and elements.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator as BinaryOperator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::*;
use crate::plan::ListPackPlan;
use crate::plan::pipeline::Blocked;
use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Step;

/// Reads list offsets, then the element range they reference, then assembles the lists and
/// keeps the selected ones.
///
/// The element range is known only once the offsets are whole, so the elements are compiled
/// then, into a new inlet.
pub(crate) struct ListPackSource {
    plan: ListPackPlan,
    /// The selection over the lists read.
    mask: Mask,
    has_validity: bool,
    /// The offsets rebased to the element range, once read, and the elements inlet.
    offsets: Option<(ArrayRef, usize)>,
    done: bool,
}

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

impl ListPackSource {
    pub(crate) fn new(plan: ListPackPlan, mask: Mask, has_validity: bool) -> Self {
        Self {
            plan,
            mask,
            has_validity,
            offsets: None,
            done: false,
        }
    }

    fn spawn_elements(&mut self, cx: &mut Cx<'_>) -> VortexResult<()> {
        let offsets_plan = self.plan.offsets()?;
        let offsets = join(offsets_plan.dtype(), drain(&mut cx.inlet(OFFSETS)))?;
        vortex_ensure!(
            offsets.len() == self.mask.len() + 1,
            "Incomplete list offsets"
        );
        let start = offsets
            .execute_scalar(0, cx.exec())?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid first list offset"))?;
        let end = offsets
            .execute_scalar(offsets.len() - 1, cx.exec())?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid last list offset"))?;
        let elements = self.plan.elements()?;
        vortex_ensure!(
            start <= end && end <= elements.row_count(),
            "List element range {start}..{end} is out of bounds"
        );
        let offsets = if start == 0 {
            offsets
        } else {
            let base = vortex_array::arrays::ConstantArray::new(start, offsets.len())
                .into_array()
                .cast(offsets.dtype().clone())?;
            offsets.binary(base, BinaryOperator::Sub)?
        };
        let inlet = cx.spawn(
            elements,
            start..end,
            Mask::new_true(usize::try_from(end - start)?),
        );
        self.offsets = Some((offsets, inlet));
        Ok(())
    }
}

impl Operator for ListPackSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::Finished);
        }
        let Some((_, elements)) = &self.offsets else {
            if !cx.inlet(OFFSETS).closed() {
                return Ok(Step::Blocked(Blocked::Inlet(OFFSETS)));
            }
            self.spawn_elements(cx)?;
            return Ok(Step::Consumed);
        };
        let elements = *elements;
        let inlets: &[usize] = if self.has_validity {
            &[VALIDITY, OFFSETS]
        } else {
            &[OFFSETS]
        };
        for &index in inlets.iter().chain([&elements]) {
            if !cx.inlet(index).closed() {
                return Ok(Step::Blocked(Blocked::Inlet(index)));
            }
        }
        let elements = join(
            self.plan.elements()?.dtype(),
            drain(&mut cx.inlet(elements)),
        )?;
        let validity = match self.plan.validity()? {
            Some(plan) => Validity::Array(join(plan.dtype(), drain(&mut cx.inlet(VALIDITY)))?),
            None => Validity::NonNullable,
        };
        let (offsets, _) = self
            .offsets
            .take()
            .ok_or_else(|| vortex_err!("Missing list offsets"))?;
        self.done = true;
        let lists = ListArray::try_new(elements, offsets, validity)?
            .into_array()
            .filter(self.mask.clone())?;
        Ok(Step::Last(lists))
    }
}

impl Source for ListPackSource {
    fn inlet_count(&self) -> usize {
        if self.has_validity { 2 } else { 1 }
    }

    fn capacity(&self, _inlet: usize) -> usize {
        UNBOUNDED
    }
}
