// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Vector;
use crate::match_each_vector;
use arrow_array::ArrayRef;
use vortex_error::VortexError;

impl TryFrom<Vector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: Vector) -> Result<Self, Self::Error> {
        match_each_vector!(value, |v| ArrayRef::try_from(v))
    }
}
