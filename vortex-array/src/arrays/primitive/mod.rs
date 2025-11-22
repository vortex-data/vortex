// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::{patch_chunk, PrimitiveArray};

mod compute;
pub use compute::{compute_is_constant, IS_CONST_LANE_WIDTH};

mod vtable;
pub use vtable::{PrimitiveMaskedValidityRule, PrimitiveVTable};

mod native_value;
pub use native_value::NativeValue;

#[cfg(test)]
mod tests;
