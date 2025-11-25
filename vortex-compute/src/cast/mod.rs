// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cast function.

use vortex_dtype::DType;
use vortex_error::VortexResult;

mod vector;

/// Trait for casting objects to different data types.
pub trait Cast {
    /// The result type after performing the cast.
    type Output;

    /// Cast the object to the specified data type.
    fn cast(&self, dtype: &DType) -> VortexResult<Self::Output>;
}
