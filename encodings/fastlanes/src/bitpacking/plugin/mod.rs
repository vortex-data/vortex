// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization plugins for the frozen bit-packed format and its external-patches adapter.

mod bitpacked;
mod patched;

#[cfg(test)]
pub(crate) use bitpacked::BitPackedMetadata;
pub use bitpacked::BitPackedPlugin;
pub(crate) use patched::BitPackedPatchedPlugin;
