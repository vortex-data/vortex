// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex string array backed by the [OnPair][onpair] short-string
//! compression library, with `cast` and `filter` pushdown.
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
mod decode;
mod kernel;
mod ops;
mod rules;
#[cfg(test)]
mod tests;

pub use array::*;
pub use compress::*;

/// Fixed token-byte over-copy width. Matches the `onpair` crate's `MAX_TOKEN_SIZE`:
/// the decoder copies exactly this many bytes per token and advances the
/// output cursor by the *true* token length. Lets the compiler emit a single
/// 128-bit SIMD store per token on x86_64 / aarch64 instead of a
/// variable-length memcpy.
pub const MAX_TOKEN_SIZE: usize = 16;
