// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::{PrimitiveArray, patch_chunk};

mod compute;
pub use compute::{IS_CONST_LANE_WIDTH, compute_is_constant};

mod vtable;
pub use vtable::{PrimitiveEncoding, PrimitiveMaskedValidityRule, PrimitiveVTable};

mod native_value;
pub use native_value::NativeValue;

#[cfg(test)]
mod tests;
