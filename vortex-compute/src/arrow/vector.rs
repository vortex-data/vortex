// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_vector::{Vector, match_each_vector};

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for Vector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        match_each_vector!(self, |v| { v.into_arrow() })
    }
}
