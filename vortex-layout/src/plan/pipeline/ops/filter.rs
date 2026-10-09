// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeping the selected rows of dense batches.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::plan::pipeline::Cx;
use crate::plan::pipeline::Input;
use crate::plan::pipeline::Operator;
use crate::plan::pipeline::Step;

/// The selected fraction of a predicate's batch at or above which it is executed whole and then
/// filtered, as the default scan's flat reader does.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Keeps the rows of `array` that `mask` selects. `predicate` says the array is a predicate's
/// result, which, mostly selected, is cheaper to execute whole than to filter lazily.
pub(crate) fn keep_selected(
    array: ArrayRef,
    mask: Mask,
    predicate: bool,
    cx: &mut Cx<'_>,
) -> VortexResult<ArrayRef> {
    if mask.all_true() {
        return Ok(array);
    }
    if predicate && mask.density() >= EXPR_EVAL_THRESHOLD {
        return array
            .execute::<Canonical>(cx.exec())?
            .into_array()
            .filter(mask);
    }
    array.filter(mask)
}

/// Keeps the selected rows of the dense batches below it, each by its own slice of the mask.
pub(crate) struct MaskStage {
    mask: Mask,
    cursor: usize,
    predicate: bool,
}

impl MaskStage {
    pub(crate) fn new(mask: Mask, predicate: bool) -> Self {
        Self {
            mask,
            cursor: 0,
            predicate,
        }
    }
}

impl Operator for MaskStage {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        match input {
            Input::Chunk(batch) => {
                let end = self.cursor + batch.len();
                vortex_ensure!(
                    end <= self.mask.len(),
                    "Filter input is longer than its mask"
                );
                let mask = self.mask.slice(self.cursor..end);
                self.cursor = end;
                Ok(Step::Last(keep_selected(batch, mask, self.predicate, cx)?))
            }
            Input::End => Ok(Step::Finished),
            Input::None => Ok(Step::Consumed),
        }
    }
}
