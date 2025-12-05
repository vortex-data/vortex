// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_vector::Vector;
use vortex_vector::match_each_vector;

use crate::arrow::IntoArrow;

impl IntoArrow for Vector {
    type Output = ArrayRef;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        match_each_vector!(self, |v| { Ok(Arc::new(v.into_arrow()?)) })
    }
}
