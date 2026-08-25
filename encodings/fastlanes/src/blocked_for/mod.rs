// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::BLOCK_SIZE;
pub use array::BlockedFoRArrayExt;
pub use array::BlockedFoRArraySlotsExt;
pub use array::BlockedFoRData;
pub use array::BlockedFoRSlots;
pub use array::blocked_for_compress::BlockSummary;
pub use array::blocked_for_compress::block_summary;

#[cfg(test)]
mod tests;

mod vtable;
pub use vtable::BlockedFoR;
pub use vtable::BlockedFoRArray;
