//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use async_trait::async_trait;
use vortex_array::IntoArray;
use vortex_array::compute::filter;
use vortex_error::VortexResult;
use vortex_expr::Scope;
use vortex_mask::Mask;

use crate::MaskEvaluation;
use crate::layouts::view::ViewEvaluation;

#[async_trait]
impl MaskEvaluation for ViewEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        let mut array = self.build_array(&mask).await?.into_array();

        let array_mask = if mask.density() < 0.2 {
            // Evaluate only the selected rows of the mask.
            array = filter(&array, &mask)?;
            let array_mask = Mask::try_from(self.expr.evaluate(&Scope::new(array))?.as_ref())?;
            mask.intersect_by_rank(&array_mask)
        } else {
            // Evaluate all rows, avoiding the more expensive rank intersection.
            array = self.expr.evaluate(&Scope::new(array))?;
            let array_mask = Mask::try_from(array.as_ref())?;
            mask.bitand(&array_mask)
        };

        println!(
            "View mask evaluation {} - {} (mask = {}) => {}",
            self.name,
            self.expr,
            mask.density(),
            array_mask.density(),
        );

        Ok(array_mask)
    }
}
