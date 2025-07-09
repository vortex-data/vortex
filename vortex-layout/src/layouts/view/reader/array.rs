//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::ArrayEvaluation;

/// A projection evaluator over a View layout.
pub(crate) struct ViewProjection {
    pub(crate) row_range: Range<u64>,
    pub(crate) expr: ExprRef,
}

#[async_trait]
impl ArrayEvaluation for ViewProjection {
    async fn invoke(&self, _mask: Mask) -> VortexResult<ArrayRef> {
        todo!()
    }
}
