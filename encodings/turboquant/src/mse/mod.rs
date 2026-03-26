// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant MSE encoding: MSE-optimal scalar quantization of rotated unit vectors.

pub use array::TurboQuantMSEArray;
pub use array::TurboQuantMSEMetadata;

pub(crate) mod array;
mod vtable;

use vortex_array::vtable::ArrayId;

/// Encoding marker type for TurboQuant MSE.
#[derive(Clone, Debug)]
pub struct TurboQuantMSE;

impl TurboQuantMSE {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant.mse");
}
