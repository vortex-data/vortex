// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant QJL encoding: inner-product-preserving quantization (MSE + QJL residual).

pub use array::TurboQuantQJLArray;
pub use array::TurboQuantQJLMetadata;

pub(crate) mod array;
mod vtable;

use vortex_array::vtable::ArrayId;

/// Encoding marker type for TurboQuant QJL.
#[derive(Clone, Debug)]
pub struct TurboQuantQJL;

impl TurboQuantQJL {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant.qjl");
}
