// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic from Vortex vector types to Arrow types.

use vortex_error::VortexResult;

mod binaryview;
mod bool;
mod decimal;
mod mask;
mod null;
mod primitive;
mod struct_;
mod vector;

/// Trait for converting Vortex vector types into Arrow types.
pub trait IntoArrow<Output> {
    /// Convert the Vortex type into an Arrow type.
    fn into_arrow(self) -> VortexResult<Output>;
}
