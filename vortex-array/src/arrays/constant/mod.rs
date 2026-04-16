// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryConstantArray;

mod array;
pub use array::ConstantData;
pub use vtable::ConstantArray;

pub(crate) mod compute;

mod vtable;

pub use vtable::Constant;
