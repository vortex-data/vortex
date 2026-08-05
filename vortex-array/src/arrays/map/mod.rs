// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical map arrays backed by [`ListView`](crate::arrays::ListView) entry storage.

mod array;
pub use array::MapArrayExt;
pub use array::MapData;
pub use array::MapDataParts;

mod vtable;
pub use vtable::Map;
pub use vtable::MapArray;

#[cfg(test)]
mod tests;
