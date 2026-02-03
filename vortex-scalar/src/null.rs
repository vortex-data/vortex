// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexExpect;

use crate::Scalar;

impl Scalar {
    /// Create a null scalar of the specified data type.
    pub fn null(dtype: DType) -> Scalar {
        Scalar::try_new(dtype, None).vortex_expect("Failed to create null scalar")
    }
}
