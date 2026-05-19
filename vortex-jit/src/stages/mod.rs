// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in stages. Lives in the same crate for v0 brevity; would split into
//! a sibling `vortex-jit-stages` crate per §9.

mod alp_decode;
mod apply_patches;
mod delta;
mod for_add;
mod load_in;
mod store_out;

pub use alp_decode::AlpDecode;
pub use apply_patches::ApplyPatchesPostLoop;
pub use delta::DeltaPrefixSum;
pub use for_add::ForAdd;
pub use load_in::LoadIn;
pub use store_out::StoreOut;
