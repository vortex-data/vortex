// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex string array backed by the [OnPair][onpair] short-string
//! compression library, with compressed-domain predicate pushdown.
//!
//! The default training preset is `dict-12` (12 bits per token, dictionary
//! capped at 4 096 entries). See [`onpair_compress`] for the entry point and
//! [`OnPairArray`] for the resulting array type.
//!
//! [onpair]: https://arxiv.org/abs/2508.02280

mod array;
mod canonical;
mod compress;
mod compute;
pub mod decode;
mod dfa;
mod kernel;
pub mod lpm;
mod ops;
mod rules;
pub mod skip;
mod slice;

/// Fixed token-byte over-copy width. Matches OnPair C++'s `MAX_TOKEN_SIZE`:
/// the decoder copies exactly this many bytes per token and advances the
/// output cursor by the *true* token length. Lets the compiler emit a single
/// 128-bit SIMD store per token on x86_64 / aarch64 instead of a
/// variable-length memcpy.
pub const MAX_TOKEN_SIZE: usize = 16;

#[cfg(test)]
mod tests;

pub use array::*;
pub use compress::*;
