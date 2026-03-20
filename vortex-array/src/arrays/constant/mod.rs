// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryConstantArray;

mod array;
pub use array::ConstantArray;

pub(crate) mod compute;

mod vtable;

pub use vtable::Constant;
pub(crate) use vtable::constant_to_dict;
pub(crate) use vtable::constant_to_run_end;
