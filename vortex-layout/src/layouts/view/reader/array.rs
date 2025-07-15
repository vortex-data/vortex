//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::filter;
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_expr::{Scope, is_root};
use vortex_mask::Mask;

use crate::ArrayEvaluation;
use crate::layouts::view::ViewEvaluation;

#[async_trait::async_trait]
impl ArrayEvaluation for ViewEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let mut array = self.build_array(&mask).await?.into_array();

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.start, self.row_range.end)?;
        }

        // Filter the array based on the row mask.
        if !mask.all_true() {
            array = filter(&array, &mask)?;
        }

        // Evaluate the projection expression.
        if !is_root(&self.expr) {
            array = self.expr.evaluate(&Scope::new(array))?;
        }

        Ok(array)
    }
}
