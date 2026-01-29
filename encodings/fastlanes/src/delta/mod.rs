// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
mod array;
mod compute;
mod vtable;

#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryDeltaArray;
pub use array::DeltaArray;
pub use array::delta_compress::delta_compress;
pub use vtable::DeltaVTable;
