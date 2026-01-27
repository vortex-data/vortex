// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::PrimitiveArray;
pub use array::PrimitiveArrayParts;
pub use array::chunk_range;
pub use array::patch_chunk;

mod compute;
pub use compute::IS_CONST_LANE_WIDTH;
pub use compute::compute_is_constant;

mod vtable;
pub use vtable::PrimitiveMaskedValidityRule;
pub use vtable::PrimitiveVTable;

mod native_value;
pub use native_value::NativeValue;

#[cfg(test)]
mod tests;
