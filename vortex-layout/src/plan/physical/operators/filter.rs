// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeping the selected rows of dense batches.

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Step;

/// The selected fraction of a predicate's batch at or above which it is executed whole and then
/// filtered, as the default scan's flat reader does.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Keeps the selected rows of the dense batches below it, each by its own slice of the mask.
pub(crate) struct MaskStage {
    mask: Mask,
    /// The row of the mask the next batch starts at.
    cursor: usize,
    /// The batches are a predicate's result, which, mostly selected, is cheaper to execute
    /// whole than to filter lazily.
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
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        let end = self.cursor + batch.len();
        vortex_ensure!(
            end <= self.mask.len(),
            "Filter input is longer than its mask"
        );
        let mask = self.mask.slice(self.cursor..end);
        self.cursor = end;
        if mask.all_true() {
            return Ok(Step::Final(batch));
        }
        if self.predicate && mask.density() >= EXPR_EVAL_THRESHOLD {
            return Ok(Step::Final(
                batch
                    .execute::<Canonical>(cx.exec())?
                    .into_array()
                    .filter(mask)?,
            ));
        }
        Ok(Step::Final(batch.filter(mask)?))
    }
}
