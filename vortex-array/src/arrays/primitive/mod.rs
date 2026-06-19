// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::PrimitiveArrayExt;
pub use array::PrimitiveData;
pub use array::PrimitiveDataParts;
pub use array::chunk_range;
pub use array::patch_chunk;
pub use vtable::PrimitiveArray;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::PrimitiveMaskedValidityRule;
pub use vtable::Primitive;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    vtable::initialize(session);
}

mod native_value;
pub use native_value::NativeValue;

#[cfg(test)]
mod tests;
