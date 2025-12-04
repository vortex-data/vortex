// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::listview::ListViewVector;
use arrow_array::ArrayRef;
use vortex_error::VortexError;

impl TryFrom<ListViewVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(_value: ListViewVector) -> Result<Self, Self::Error> {
        todo!("Figure out how to do this")
    }
}
