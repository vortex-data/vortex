// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_vector::listview::ListViewVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for ListViewVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        todo!("Figure out how to do this")
    }
}
