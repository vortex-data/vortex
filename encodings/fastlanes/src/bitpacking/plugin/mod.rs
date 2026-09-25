// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`vortex_array::ArrayPlugin`]s for bit-packed arrays.
//!
//! [`BitPackedPlugin`] owns the wire history of `BitPacked`: the frozen `fastlanes.bitpacked`
//! format for arrays whose chunks share one width, and `fastlanes.bitpacked_v2`, whose width
//! table and offsets children describe each chunk. [`BitPackedPatchedPlugin`] reads both and lifts
//! interior patches into a `Patched` array.

mod bitpacked;
mod patched;

#[cfg(test)]
pub(crate) use bitpacked::BitPackedMetadata;
pub use bitpacked::BitPackedPlugin;
#[cfg(test)]
pub(crate) use bitpacked::BitPackedV2Metadata;
pub use bitpacked::bitpacked_v2_id;
pub(crate) use patched::BitPackedPatchedPlugin;
