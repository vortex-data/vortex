// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryRLEArray;

mod array;
pub use array::RLEArray;

mod compute;
mod kernel;

mod vtable;
pub use vtable::RLEVTable;
