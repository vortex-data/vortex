// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryConstantArray;

mod array;
pub use array::ConstantArray;
pub(crate) use array::ConstantMetadata;
pub(crate) use vtable::canonical::constant_canonicalize;

pub(crate) mod compute;

mod vtable;

pub use vtable::ConstantVTable;
