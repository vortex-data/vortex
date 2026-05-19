// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in stages. Lives in the same crate for v0 brevity; would split into
//! a sibling `vortex-jit-stages` crate per §9.

mod alp_decode;
mod apply_patches;
mod bitpacked;
mod delta;
mod dict;
mod for_add;
mod load_in;
mod rle;
mod store_out;

pub use alp_decode::AlpDecode;
pub use apply_patches::ApplyPatchesPostLoop;
pub use bitpacked::{BitPackedLoad, pack_dense, unpack_one};
pub use delta::DeltaPrefixSum;
pub use dict::DictLookup;
pub use for_add::ForAdd;
pub use load_in::LoadIn;
pub use rle::RleExpandPostLoop;
pub use store_out::StoreOut;
